import os.path
import numpy as np
from Generals.ScannerGenerals import ScannerOption
from Generals.PointSpreadFunction import PointSpreadFunction
from Generals.TOFGenerals import TOFOption


class ReconOption:
    def __init__(self, img_dim: np.array, voxel_size: np.array, ex_cdf_path: str, tx_cdf_path: str, bx_cdf_path: str, output_dir: str, scanner_option: ScannerOption, psf_option: PointSpreadFunction, num_of_iterations, num_of_subsets, device_id):
        self.img_dim = img_dim
        self.voxel_size = voxel_size
        self.img_origin = -(img_dim / 2 - 0.5) * voxel_size
        self.num_of_iterations = num_of_iterations
        self.num_of_subsets = num_of_subsets
        self.output_dir = output_dir
        self.ex_cdf_path = ex_cdf_path
        self.tx_cdf_path = tx_cdf_path
        self.bx_cdf_path = bx_cdf_path
        self.scanner_option = scanner_option
        self.psf_option = psf_option
        self.output_paths = self.construct_output_path()
        self.device = device_id
        self.scan_type = "emission" if ex_cdf_path is not None else "transmission"

    def construct_output_path(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        path_dict = {
            "sense_img_output_path": os.path.join(self.output_dir, "sensetivity_img.raw"),
            "lin_int_proj_output_path": os.path.join(self.output_dir, "linear_integral_projection.raw"),
            "ex_img_output_path": os.path.join(self.output_dir, "emission_recon_%di%ds_psf_%s.raw" % (self.num_of_iterations, self.num_of_subsets, "on" if self.psf_option is not None else "off")),
            "tx_img_output_path": os.path.join(self.output_dir, "transmission_recon_%di%ds_psf_%s.raw" % (self.num_of_iterations, self.num_of_subsets, "on" if self.psf_option is not None else "off")),
            "console_output_path": os.path.join(self.output_dir, "console_output.raw"),
            "blank_sinogram_output_path": os.path.join(self.output_dir, "blank_sinogram.raw"),
            "transmission_sinogram_output_path": os.path.join(self.output_dir, "transmission_sinogram.raw"),

        }
        return path_dict

    def return_details(self):
        details = [
            "===== Reconstruction Options =====\n",
            "img_dim = [{}, {}, {}]\n".format(self.img_dim[0], self.img_dim[1], self.img_dim[2]),
            "voxel_size = [{}, {}, {}]\n".format(self.voxel_size[0], self.voxel_size[1], self.voxel_size[2]),
            "img_origin = [{}, {}, {}]\n".format(self.img_origin[0], self.img_origin[1], self.img_origin[2]),
            "iterations = {}\n".format(self.num_of_iterations),
            "subsets = {}\n".format(self.num_of_subsets),
        ]
        return details

    def get_lor_location(self, start_ids, end_ids):
        start_ids = start_ids.astype(int)
        end_ids = end_ids.astype(int)
        pos_comb = np.concatenate((
            self.scanner_option.crystals_position[start_ids, :],
            self.scanner_option.crystals_position[end_ids, :]
        ), axis=1)
        return pos_comb

    def get_coins(self, cdf_path, start_index, end_index=None):
        current_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
        block_size = np.empty(0, dtype=current_dtype).itemsize
        event_num = os.path.getsize(cdf_path) / block_size
        if end_index is None or end_index > event_num:
            end_index = event_num

        start_index = int(start_index)
        end_index = int(end_index)
        with open(cdf_path, 'rb') as f:
            f.seek(block_size * start_index)
            data = f.read(block_size * (end_index - start_index))

        coins = np.frombuffer(data, dtype=current_dtype)
        coins = np.column_stack((coins["castor_id_1"], coins["castor_id_2"]))

        # 这里为了避免后续有重复的 id 相反的 LOR，决定把小 id 放在 id1 ，大 id 放 id2
        # 在后续读取 tof 时要注意将 tof 置反
        flip_flag = coins[:, 0] > coins[:, 1]
        coins[flip_flag, 0], coins[flip_flag, 1] = coins[flip_flag, 1].copy(), coins[flip_flag, 0].copy()

        coins_1d = coins[:, 0] * self.scanner_option.crystal_num + coins[:, 1]
        uni_coins, uni_counts = np.unique(coins_1d, return_counts=True, axis=0)
        coins_id_1 = uni_coins // self.scanner_option.crystal_num
        coins_id_2 = uni_coins % self.scanner_option.crystal_num
        coins = np.concatenate((
            coins_id_1.reshape(-1, 1), coins_id_2.reshape(-1, 1), uni_counts.reshape(-1, 1)
        ), axis=1)

        # coins = [id_1, id2, counts]
        return coins

    def get_coins_wtof(self, cdf_path, tof_option: TOFOption, scanner_option: ScannerOption, start_index, end_index=None, return_with_full_lor=False):
        print("Loading cdf...")
        current_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
        block_size = np.empty(0, dtype=current_dtype).itemsize
        event_num = os.path.getsize(cdf_path) / block_size
        if end_index is None or end_index > event_num:
            end_index = event_num

        start_index = int(start_index)
        end_index = int(end_index)
        with open(cdf_path, 'rb') as f:
            f.seek(block_size * start_index)
            data = f.read(block_size * (end_index - start_index))

        coins = np.frombuffer(data, dtype=current_dtype)
        castor_id_1 = coins["castor_id_1"].astype(np.uint64)
        castor_id_2 = coins["castor_id_2"].astype(np.uint64)
        tof = coins["tof"].astype(np.float32)
        # 这里为了避免后续有重复的 id 相反的 LOR，决定把小 id 放在 id1 ，大 id 放 id2
        # 在后续读取 tof 时要注意将 tof 置反
        flip_flag = castor_id_1 > castor_id_2
        castor_id_1[flip_flag], castor_id_2[flip_flag], tof[flip_flag] = castor_id_2[flip_flag].copy(), castor_id_1[flip_flag].copy(), -tof[flip_flag].copy()

        tof_bin_index, rm_option = tof_option.get_tof_bin_index(tof)

        # if remove coins out of tof range
        tof_bin_index = tof_bin_index[~rm_option]
        castor_id_1, castor_id_2, tof = castor_id_1[~rm_option], castor_id_2[~rm_option], tof[~rm_option]

        if return_with_full_lor:
            rows, cols = np.triu_indices(scanner_option.crystal_num)
            uni_coins = np.column_stack((rows, cols))
            pair_indices = (castor_id_1 * (2 * scanner_option.crystal_num - castor_id_1 + 1)) // 2 + (castor_id_2 - castor_id_1)
            flat_indices = pair_indices * tof_option.tof_bin_num + tof_bin_index
            counts = np.bincount(flat_indices.astype(int), minlength=uni_coins.shape[0] * tof_option.tof_bin_num)
            histo = counts.reshape((uni_coins.shape[0], tof_option.tof_bin_num)).astype(np.uint32)
        else:
            coins_1d = castor_id_1 * self.scanner_option.crystal_num + castor_id_2
            uni_coins, uni_row_index = np.unique(coins_1d, return_inverse=True)
            coins_id_1 = (uni_coins // self.scanner_option.crystal_num).astype(np.uint32)
            coins_id_2 = (uni_coins % self.scanner_option.crystal_num).astype(np.uint32)
            uni_coins = np.column_stack((coins_id_1, coins_id_2))
            histo = np.zeros([uni_coins.shape[0], tof_option.tof_bin_num], dtype=np.uint32)
            np.add.at(histo, tuple(np.column_stack((uni_row_index, tof_bin_index)).astype(np.uint32).T), 1)

        # coins = [id_1, id2], histo = [n_events, tof_bin]
        return uni_coins.astype(int), histo



