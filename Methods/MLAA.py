import numpy as np
import os
from tqdm import tqdm
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption
from Generals.Projector import Projector
from Generals.TOFGeneranls import TOFOption
from Generals.PointSpreadFunction import PointSpreadFunction
import matplotlib.pyplot as plt
from temp.functions_from_kernal_MLAA import trl_curvature


class MLAA(ReconOption):
    def __init__(self, img_dim: np.array, voxel_size: np.array, output_dir: str, scanner_option: ScannerOption, psf_option: PointSpreadFunction, tof_option: TOFOption, ac_map: np.array, mu_map: np.array, ex_cdf_path=None, device_id=0):
        super().__init__(img_dim=img_dim, voxel_size=voxel_size, ex_cdf_path=ex_cdf_path, output_dir=output_dir, scanner_option=scanner_option, psf_option=psf_option, num_of_iterations=0, num_of_subsets=0, device_id=0, tx_cdf_path=None, bx_cdf_path=None)
        self.projector = Projector(self, scanner_option, psf_option, tof_option, device_id)
        self.ac_map = ac_map
        self.mu_map = mu_map / 10
        self.tof_option = tof_option
        self.sense_img = None
        self.measurement = None
        self.yi = None
        self.alpha_p = 0.1
        self.n = 170

    def get_sense_img(self):
        self.sense_img = np.zeros(self.img_dim)
        for i in tqdm(range(self.scanner_option.crystal_per_layer), desc="calculating sensesitivity image..."):
            ai = self.projector.projection_forward_lors_wtof(self.mu_map, np.ones([self.scanner_option.crystal_per_layer - i]) * i, np.arange(i, self.scanner_option.crystal_per_layer), False)
            ai = np.exp(-ai)
            self.sense_img += self.projector.projection_backward_wtof(
                start_index=np.ones([self.scanner_option.crystal_per_layer - i]) * i,
                end_index=np.arange(i, self.scanner_option.crystal_per_layer),
                sinogram=np.ones([self.scanner_option.crystal_per_layer - i]),
                add_psf=False
            )
        self.sense_img.tofile(self.output_dir + "/sense_img.raw")

    def set_0_outsize_fov(self, img):
        h, w = img.shape[:2]  # 获取图像高度和宽度（忽略通道数）
        cx, cy = w // 2, h // 2  # 图像中心坐标（x轴：宽度方向，y轴：高度方向）

        x = np.arange(w)  # x轴坐标：0 ~ w-1
        y = np.arange(h)  # y轴坐标：0 ~ h-1
        xx, yy = np.meshgrid(x, y)  # 生成网格：xx.shape=(h,w), yy.shape=(h,w)
        distance = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        mask = (distance <= self.scanner_option.scanner_radius).astype(int)

        # 3D图像（彩色图）需要扩展掩码维度（从2D->3D），与图像通道匹配
        if len(img.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)  # mask.shape=(h,w,1)，广播到通道数

        result = img * mask.astype(img.dtype)  # 掩码转图像数据类型（避免类型不匹配）
        return result

    def get_ac_map_update(self):
        ac_fwd = self.projector.projection_forward_lors_wtof(self.ac_map, self.measurement[:, 1], self.measurement[:, 2], False, 1)
        ac_update = self.projector.projection_backward_wtof(
            start_index=self.measurement[:, 1],
            end_index=self.measurement[:, 2],
            sinogram=(self.yi / ac_fwd),
            add_psf=False
        )
        # self.get_sense_img()
        self.ac_map = self.ac_map / self.sense_img * ac_update
        self.ac_map[self.sense_img == 0] = 0
        np.nan_to_num(self.ac_map, copy=False, nan=1e-8, posinf=1e-8, neginf=1e-8)
        self.ac_map = self.set_0_outsize_fov(self.ac_map)


    def get_mu_map_update(self):
        bi = self.projector.projection_forward_lors_wtof(self.ac_map, self.measurement[:, 1], self.measurement[:, 2], False)
        ai = np.exp(-self.projector.projection_forward_lors_wtof(self.mu_map, self.measurement[:, 1], self.measurement[:, 2], False))
        bp_aibi = self.projector.projection_backward_wtof(self.measurement[:, 1], self.measurement[:, 2], (ai * bi), False)
        bp_yi = self.projector.projection_backward_wtof(self.measurement[:, 1], self.measurement[:, 2], self.yi, False)
        mu_update = (self.alpha_p / self.n) * (1 - bp_yi / bp_aibi)
        np.nan_to_num(mu_update, copy=False, nan=1e-8, posinf=1e-8, neginf=1e-8)
        self.mu_map += mu_update
        self.mu_map[self.mu_map < 0] = 1e-8
        self.mu_map = self.set_0_outsize_fov(self.mu_map)

    def get_attn_ml_sps(self):
        # from kernel MLAA
        bi = self.projector.projection_forward_lors(self.ac_map, self.measurement[:, 0], self.measurement[:, 1],
                                                    False)
        ai = np.exp(
            -self.projector.projection_forward_lors(self.mu_map, self.measurement[:, 0], self.measurement[:, 1],
                                                    False))
        yb = ai * bi
        yt = self.measurement[:, 2]
        yr = (1 - yt / yb) * yb
        gx = self.projector.projection_backward(np.column_stack((
            self.measurement[:, 0], self.measurement[:, 1], yr
        )), False)
        nt = trl_curvature(yt, bi, 0, bi, 'oc')
        aa = self.projector.projection_forward_lors(np.ones_like(self.ac_map), self.measurement[:, 0],
                                                    self.measurement[:, 1], False)
        wx = self.projector.projection_backward(np.column_stack((
            self.measurement[:, 0], self.measurement[:, 1], nt * aa
        )), False)

        self.mumap = self.mumap + gx / wx
        self.mu_map[wx == 0] = 0

    def get_objection_function_score(self):
        ai = self.projector.projection_forward_lors(self.mu_map, self.measurement[:, 0], self.measurement[:, 1], False)
        bi = np.exp(-self.projector.projection_forward_lors(self.ac_map, self.measurement[:, 0], self.measurement[:, 1], False))
        ri = ai * bi
        score = np.sum(-ri + self.measurement[:, 2] * np.log(ri))
        return score

    def run(self):
        iteration = 50
        self.measurement = self.get_coins_wtof(self.ex_cdf_path, tof_option, 0)
        self.yi = np.ones([self.measurement.shape[0], self.tof_option.tof_bin_num], dtype=np.float32) * 1e-8
        self.yi[np.arange(self.measurement.shape[0], dtype=int), self.measurement[:, 0]] = self.measurement[:, 3]
        self.get_sense_img()
        obj_func_score = np.zeros(iteration)
        for i in range(iteration):
            self.get_ac_map_update()
            # self.get_mu_map_update()
            self.ac_map.astype(np.float32).tofile(self.output_dir + r"\ac_map_it%d.raw" % i)
            # self.mu_map.astype(np.float32).tofile(self.output_dir + r"\mu_map_it%d.raw" % i)
            obj_func_score[i] = self.get_objection_function_score()

            # plt.plot(obj_func_score)
            # plt.scatter(np.arange(iteration), obj_func_score, s=30, marker="*")
            # plt.savefig(r"D:\BaiduNetdiskDownload\data_for_mlaa\test_output\tof_mlaa\score.jpg", bbox_inches="tight")
        obj_func_score.astype(np.float32).tofile(r"D:\BaiduNetdiskDownload\data_for_mlaa\test_output\tof_mlem_noac\score.raw")



if __name__ == "__main__":
    os.chdir(r"D:\github_code\pet-reconstruction")
    scanner_option = ScannerOption("WBBrain_20251219")
    ac_map = np.ones([170, 170, 170])
    mu_map = np.fromfile(r"D:\BaiduNetdiskDownload\data_for_mlaa\clear_mumap_dim170.raw", dtype=np.float32).reshape(170, 170, 170).transpose([2, 1, 0])
    psf_option = PointSpreadFunction(sigma=1)
    tof_option = TOFOption(tof_resolution=300, tof_bin_num=21, tof_bin_width_in_ps=100)
    mlaa = MLAA(
        img_dim=np.array([170, 170, 170]),
        voxel_size=np.array([1, 1, 1]),
        output_dir=r"D:\BaiduNetdiskDownload\data_for_mlaa\test_output\tof_mlem_noac",
        scanner_option=scanner_option,
        ex_cdf_path=r"D:\BaiduNetdiskDownload\data_for_mlaa\trues.cdf",
        psf_option=psf_option,
        device_id=0,
        ac_map=ac_map,
        mu_map=mu_map,
        tof_option=tof_option
    )
    mlaa.run()