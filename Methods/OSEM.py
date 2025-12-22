import numpy as np
import array_api_compat.cupy as xp
import os
import struct
from tqdm import tqdm
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption
from Methods.Projector import Projector
from Methods.PointSpreadFunction import PointSpreadFunction
from Conversions.castor_id_to_sinogram import get_sinogram, check_id_in_sinogram
from multiprocessing import Pool, Array


class OSEM(ReconOption):
    def __init__(self, img_dim: np.array, voxel_size: np.array, output_dir: str, scanner_option: ScannerOption, psf_option: PointSpreadFunction, use_ssrb=True, ex_cdf_path=None, tx_cdf_path=None, bx_cdf_path=None, num_of_iterations=5, num_of_subsets=4, device_id=0):
        super().__init__(img_dim, voxel_size, ex_cdf_path, tx_cdf_path, bx_cdf_path, output_dir, scanner_option, psf_option, num_of_iterations, num_of_subsets, device_id)
        self.projector = Projector(self, scanner_option, psf_option, device_id)
        self.estimate = np.ones(self.img_dim)
        self.use_ssrb = use_ssrb
        self.sense_img = None
        self.lin_int_proj = None
        if self.scan_type == "transmission":
            self.get_linear_integral_projection()
        else:
            self.get_sense_img()

    def get_sense_img(self):
        if os.path.exists(self.output_paths["sense_img_output_path"]):
            print("Sensetivity image is found in save path. ")
            self.sense_img = np.fromfile(self.output_paths["sense_img_output_path"], dtype=np.float32).reshape(self.img_dim)
        else:
            self.sense_img = np.zeros(self.img_dim)
            for i in tqdm(range(self.scanner_option.crystal_per_layer), desc="calculating sensesitivity image..."):
                self.sense_img += self.projector.projection_backward(np.concatenate((
                    np.ones([self.scanner_option.crystal_per_layer - i, 1]) * i,
                    np.arange(i, self.scanner_option.crystal_per_layer).reshape(-1, 1),
                    np.ones([self.scanner_option.crystal_per_layer - i, 1])
                ), axis=1), False)
            self.sense_img.astype(np.float32).tofile(self.output_paths["sense_img_output_path"])

    def get_emission_update(self, subset_index):
        total_counts = os.path.getsize(self.ex_cdf_path) / struct.calcsize("III")
        subset_size = total_counts / self.num_of_subsets

        start_index = subset_index * subset_size
        end_index = total_counts if subset_index == self.num_of_subsets - 1 else (subset_index + 1) * subset_size

        # 将 estimate 前投影与 measured 计算误差，在反投影回图像域做更新
        coins = self.get_coins(self.ex_cdf_path, start_index, end_index)
        estimate_fwd = self.projector.projection_forward_lors(self.estimate, coins[:, 0], coins[:, 1])

        error_fwd = coins[:, 2] / estimate_fwd
        np.nan_to_num(error_fwd, copy=False, nan=0, posinf=0, neginf=0)

        error_img = self.projector.projection_backward(np.concatenate((
                coins[:, :2],
                error_fwd.reshape(-1, 1)
        ), axis=1), add_psf=True)

        self.estimate = self.estimate / self.sense_img * error_img
        np.nan_to_num(self.estimate, copy=False, nan=0, posinf=0, neginf=0)

    def get_transmission_update(self, subset_index):
        max_phi = self.lin_int_proj[:, 3].max() + 1

        # 方法1：等间隔角度重建（需要根据实际的 LOR 计算 sensetivity image）
        phi_index = np.arange(0, max_phi, self.num_of_subsets) + subset_index
        measured_proj = self.lin_int_proj[np.isin(self.lin_int_proj[:, 3], phi_index), :3]
        sense_img = self.projector.projection_backward(np.column_stack((
                    measured_proj[:, 0],
                    measured_proj[:, 1],
                    np.ones([measured_proj.shape[0]])
                )), False)

        estimate_fwd = self.projector.projection_forward_lors(self.estimate, measured_proj[:, 0], measured_proj[:, 1], add_psf=False)

        error_fwd = measured_proj[:, 2] / estimate_fwd
        np.nan_to_num(error_fwd, copy=False, nan=0, posinf=0, neginf=0)

        error_img = self.projector.projection_backward(np.concatenate((
            measured_proj[:, :2],
            error_fwd.reshape(-1, 1)
        ), axis=1), add_psf=False)

        self.estimate = self.estimate * (error_img / sense_img)
        np.nan_to_num(self.estimate, copy=False, nan=0, posinf=0, neginf=0)

    def get_linear_integral_projection(self, n_splits=60):
        # n 行 4 列 （id_0 -- id_1 -- counts -- phi）（只保留非零值）
        if os.path.exists(self.output_paths["lin_int_proj_output_path"]):
            print("Linear integral projection is found in save path. ")
            self.lin_int_proj = np.fromfile(self.output_paths["lin_int_proj_output_path"], dtype=np.float32).reshape(-1, 4)
        else:
            tx_sinogram = get_sinogram(self.scanner_option)[self.use_ssrb]
            bx_sinogram = get_sinogram(self.scanner_option)[self.use_ssrb]
            for split in tqdm(range(n_splits)):
                current_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
                block_size = np.empty(0, dtype=current_dtype).itemsize
                tx_size = os.path.getsize(self.tx_cdf_path) / block_size
                bx_size = os.path.getsize(self.bx_cdf_path) / block_size
                tx_start, bx_start = np.ceil(tx_size / n_splits) * split, np.ceil(bx_size / n_splits) * split
                tx_end, bx_end = np.ceil(tx_size / n_splits) * (split + 1), np.ceil(bx_size / n_splits) * (split + 1)
                tx_coins = self.get_coins(self.tx_cdf_path, start_index=tx_start, end_index=tx_end)
                bx_coins = self.get_coins(self.bx_cdf_path, start_index=bx_start, end_index=bx_end)
                tx_sinogram += get_sinogram(self.scanner_option, coins=tx_coins, get_ssrb=self.use_ssrb)[self.use_ssrb]
                bx_sinogram += get_sinogram(self.scanner_option, coins=bx_coins, get_ssrb=self.use_ssrb)[self.use_ssrb]

            tx_sinogram.astype(np.float32).tofile(self.output_paths["transmission_sinogram_output_path"])
            bx_sinogram.astype(np.float32).tofile(self.output_paths["blank_sinogram_output_path"])

            # 异常值处理
            measured_proj = np.log(bx_sinogram / tx_sinogram)
            np.nan_to_num(measured_proj, copy=False, nan=0, posinf=0, neginf=0)
            measured_proj[measured_proj < 0] = 0
            self.lin_int_proj = check_id_in_sinogram(self.scanner_option, measured_proj)
            self.lin_int_proj.astype(np.float32).tofile(self.output_paths["lin_int_proj_output_path"])

    def run(self):
        if self.scan_type == "emission":
            for iteration in range(self.num_of_iterations):
                for subset in range(self.num_of_subsets):
                    self.get_emission_update(subset)
                self.estimate.astype(np.float32).tofile(self.output_paths["ex_img_output_path"])
        elif self.scan_type == "transmission":
            for iteration in range(self.num_of_iterations):
                for subset in range(self.num_of_subsets):
                    self.get_transmission_update(subset)
                self.estimate.astype(np.float32).tofile(self.output_paths["tx_img_output_path"])
        else:
            raise Exception("Wrong scan type input. ")


if __name__ == "__main__":
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction/")
    scanner_option = ScannerOption(
        "PET_FOR_BRAIN_11panel_DOI0",
    )
    psf_option = PointSpreadFunction(sigma=1)
    tx_osem = OSEM(
        img_dim=np.array([170, 170, 170]),
        voxel_size=np.array([1, 1, 1]),
        output_dir=r"path/to/output",
        scanner_option=scanner_option,
        # ex_cdf_path=None,
        tx_cdf_path=r"path/to/tx_cdf",
        bx_cdf_path=r"path/to/bx_cdf",
        num_of_iterations=4,
        num_of_subsets=3,
        use_ssrb=False,
        psf_option=None,
        device_id=1
    )

    tx_osem.run()
    del tx_osem

    tx_osem = OSEM(
        img_dim=np.array([170, 170, 170]),
        voxel_size=np.array([1, 1, 1]),
        output_dir=r"path/to/output",
        scanner_option=scanner_option,
        # ex_cdf_path=None,
        tx_cdf_path=r"path/to/tx_cdf",
        bx_cdf_path=r"path/to/bx_cdf",
        num_of_iterations=4,
        num_of_subsets=3,
        use_ssrb=True,
        psf_option=None,
        device_id=1
    )

    tx_osem.run()

