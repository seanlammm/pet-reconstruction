import numpy as np
import struct
import os
from tqdm import tqdm
import sys
import warnings

sys.path.append(r"/home/linyuejie/Documents/codes/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Conversions.coin_dat_to_info import coin_dat_to_info, delayed_coin_dat_to_info, gate_coin_dat_to_info
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram

class CoinsToCdf:
    def __init__(self, scanner_option: ScannerOption, file_path, save_path, save_filename, total_time, time_window, with_random_rate: bool = False, with_attenuation_correction_factor: bool = False, with_normalization_factor: bool = True, with_scatter_correction_factor: bool = False, with_doi: bool = False, with_tof: bool = False, tof_resolution=0, ac_factor_file_path=None, norm_factor_file_path=None, sc_factor_file_path=None, with_multiple_coins=False, with_energy_window=None):
        self.scanner_option = scanner_option
        self.file_path = file_path
        self.save_path = save_path
        self.save_filename = save_filename
        self.total_time = np.array([total_time]).reshape(-1)
        self.tw = time_window
        self.w_rr = with_random_rate
        self.w_acf = with_attenuation_correction_factor
        self.w_norm = with_normalization_factor
        self.w_sc = with_scatter_correction_factor
        self.w_doi = with_doi
        self.w_tof = with_tof
        self.tof_resolution = tof_resolution
        self.path_to_acf = ac_factor_file_path
        self.path_to_normf = norm_factor_file_path
        self.path_to_scf = sc_factor_file_path
        self.w_multi = with_multiple_coins
        self.w_engwindow = with_energy_window
        self.tof_range = np.zeros(2)

        if isinstance(total_time, int):
            self.time_loop = np.arange(1, self.total_time + 1)
        else:
            self.time_loop = self.total_time
        self.total_time = self.time_loop.shape[0]

        # energy_window check
        if self.w_engwindow[0] % 10 != 0 or self.w_engwindow[1] % 10 != 0:
            self.w_engwindow[0] = self.w_engwindow[0] // 10 * 10
            self.w_engwindow[1] = self.w_engwindow[1] // 10 * 10
            print("Energy window will be changed to [%d, %d]" % (self.w_engwindow[0], self.w_engwindow[1]))

        if self.w_rr:
            self.sum_singles_rates = self.get_randoms_rate()
        if self.w_acf:
            self.ac_factor = self.get_ac_factor()
        if self.w_norm:
            self.norm_factor = self.get_norm_factor()
        if self.w_sc:
            self.scatter_factor = self.get_scatter_factor()

    def get_delayed_random_rate(self, coins_data=None):
        if coins_data is None:
            n_rings = self.scanner_option.rings
            n_det = self.scanner_option.crystal_per_ring
            s_width = n_det - self.scanner_option.submodules_xy * self.scanner_option.crystals_per_submodule
            delayed_sinogram = np.zeros([n_rings, n_rings, n_det//2, s_width], dtype=np.float32)
            for time in tqdm(self.time_loop, desc="Collecting delayed counts..."):
                for pack in range(1, 300):
                    delayed_coins_file = os.path.join(self.file_path, "delayed_coins%d_%d.dat" % (time, pack))
                    if not os.path.exists(delayed_coins_file):
                        continue
                    uni_ids, counts = delayed_coin_dat_to_info(delayed_coins_file, with_multiple=self.w_multi, tw=self.tw, eng_window=self.w_engwindow)
                    sino_index, rm_index = castor_id_to_sinogram(self.scanner_option, uni_ids)
                    delayed_sinogram[sino_index[:, 0], sino_index[:, 1], sino_index[:, 2], sino_index[:, 3]] += counts[rm_index == 0].astype(np.float32)
                    sino_index, rm_index = castor_id_to_sinogram(self.scanner_option, uni_ids)
                    delayed_sinogram[sino_index[:, 0], sino_index[:, 1], sino_index[:, 2], sino_index[:, 3]] += counts[rm_index == 0].astype(np.float32)

            total_time = self.total_time * 60  # unit: second
            delayed_sinogram /= total_time

            return delayed_sinogram
        else:
            sino_index, rm_index = castor_id_to_sinogram(self.scanner_option, coins_data)
            random_rate = np.zeros_like(rm_index, dtype=np.float32)
            random_rate[rm_index == 0] = self.sum_singles_rates[sino_index[:, 0], sino_index[:, 1], sino_index[:, 2], sino_index[:, 3]]

            return random_rate

    def get_randoms_rate(self, coins_data=None):
        if coins_data is None:
            num_of_crystal = self.scanner_option.crystal_per_layer * self.scanner_option.layers
            sum_singles_counts = np.zeros([num_of_crystal])
            for time in tqdm(self.time_loop, desc="Collecting singles counts"):
                for pack in range(1, 300):
                    singles_counts_file = os.path.join(self.file_path, "singles_counts%d_%d.dat" % (time, pack))
                    if not os.path.exists(singles_counts_file):
                        continue

                    item_size = np.dtype(np.uint16).itemsize
                    start_idx = int(self.w_engwindow[0] // 10)
                    end_idx = int(self.w_engwindow[1] // 10)
                    counts = (end_idx - start_idx) * num_of_crystal  # +1 代表包含右边界
                    offset = start_idx * num_of_crystal * item_size
                    singles_counts = np.fromfile(singles_counts_file, dtype=np.uint16, count=counts, offset=offset).reshape([-1, num_of_crystal])
                    sum_singles_counts += np.sum(singles_counts, axis=0)

            total_time = self.total_time * 60  # unit: second 11panel
            # total_time = self.total_time * 6  # unit: second  Cardiac
            # total_time = self.total_time * 5  # unit: second  TotalBody
            sum_singles_rates = sum_singles_counts / total_time
            return sum_singles_rates

        else:
            # coin_time_window = self.tw * 2.5 * 1e-9  # time_window * seconds per time window
            coin_time_window = self.tw * 1e-12  # time_window * seconds per time window
            singles_rate_0 = self.sum_singles_rates[coins_data[:, 0]]
            singles_rate_1 = self.sum_singles_rates[coins_data[:, 1]]
            randoms_rate = 2 * coin_time_window * singles_rate_0 * singles_rate_1

            return randoms_rate.astype(np.float32)

    # def get_randoms_rate(self, coins_data=None): # total-body
    #     if coins_data is None:
    #         num_of_crystal = self.scanner_option.crystal_per_layer * self.scanner_option.layers
    #         sum_singles_counts = np.zeros([num_of_crystal])
    #         for time in tqdm(self.time_loop, desc="Collecting singles counts"):
    #             for pack in range(1, 300):
    #                 singles_counts_file = os.path.join(self.file_path, "singles_counts%d_%d.dat" % (time, pack))
    #                 if not os.path.exists(singles_counts_file):
    #                     continue
    #
    #                 item_size = np.dtype(np.uint16).itemsize
    #                 start_idx = int(self.w_engwindow[0] // 10)
    #                 end_idx = int(self.w_engwindow[1] // 10)
    #                 counts = (end_idx - start_idx) * num_of_crystal  # +1 代表包含右边界
    #                 offset = start_idx * num_of_crystal * item_size
    #                 singles_counts = np.fromfile(singles_counts_file, dtype=np.uint16, count=counts, offset=offset).reshape([-1, num_of_crystal])
    #                 sum_singles_counts += np.sum(singles_counts, axis=0)
    #
    #         total_time = self.total_time * 5  # unit: second  TotalBody
    #         sum_singles_rates = sum_singles_counts / total_time
    #         return sum_singles_rates
    #
    #     else:
    #         # coin_time_window = self.tw * 2.5 * 1e-9  # time_window * seconds per time window
    #         coin_time_window = self.tw * 1e-12  # time_window * seconds per time window
    #         singles_rate_0 = self.sum_singles_rates[coins_data[:, 0] + (22*12*10*6*8)]
    #         singles_rate_1 = self.sum_singles_rates[coins_data[:, 1] + (22*12*10*6*8)]
    #         randoms_rate = 2 * coin_time_window * singles_rate_0 * singles_rate_1
    #
    #         return randoms_rate.astype(np.float32)


    def get_ac_factor(self, coins_data=None):
        if coins_data is None:
            ac_factor = np.fromfile(self.path_to_acf, dtype=np.float32)
            return ac_factor
        else:
            swap_flag = coins_data[:, 0] > coins_data[:, 1]
            swap_value = coins_data[swap_flag, 0]
            coins_data[swap_flag, 0] = coins_data[swap_flag, 1]
            coins_data[swap_flag, 1] = swap_value

            index = ((self.scanner_option.crystal_per_layer + self.scanner_option.crystal_per_layer - coins_data[:, 0] + 1) * coins_data[:, 0] / 2 + coins_data[:, 1] - coins_data[:, 0]).astype(int)
            coins_ac_factor = self.ac_factor[index]
            return coins_ac_factor

    def get_norm_factor(self, coins_data=None):
        if coins_data is None:
            norm_factor = np.fromfile(self.path_to_normf, dtype=np.float32)
            return norm_factor
        else:
            coins_data = coins_data.astype(np.uint32)
            index = (self.scanner_option.crystal_per_layer * coins_data[:, 0] + coins_data[:, 1]).astype(np.uint32)
            coins_norm_factor = self.norm_factor[index]
            return coins_norm_factor

    def get_scatter_factor(self, coins_data=None):
        if coins_data is None:
            scatter_factor = np.fromfile(self.path_to_scf, dtype=np.float32)
            return scatter_factor
        else:
            coins_data = coins_data.astype(np.uint32)
            index = (self.scanner_option.crystal_per_layer * coins_data[:, 0] + coins_data[:, 1]).astype(np.uint32)
            coins_sc_factor = self.scatter_factor[index]
            return coins_sc_factor

    def run(self):
        if self.w_rr:
            self.save_filename += "_wrr"
        if self.w_acf:
            self.save_filename += "_wacf"
        if self.w_sc:
            self.save_filename += "_wsc"
        if self.w_norm:
            self.save_filename += "_wnorm"
        if self.w_doi:
            self.save_filename += "_wdoi"
        if self.w_tof:
            self.save_filename += "_wtof"
        if self.w_engwindow is not None:
            self.save_filename += "_wengwin_%d_%d" % (self.w_engwindow[0], self.w_engwindow[1])
        if not self.w_multi:
            self.save_filename += "_womulti"

        cdf_file = os.path.join(self.save_path, self.save_filename + ".cdf")
        cdh_file = os.path.join(self.save_path, self.save_filename + ".cdh")
        console_output_file = os.path.join(self.save_path, "console_" + self.save_filename + ".txt")
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if os.path.exists(cdf_file) or os.path.exists(console_output_file) or os.path.exists(cdh_file):
            print("File already exists.\nProgram will do nothing but return. ")
            return

        with open(console_output_file, 'a') as file:
            print(f"Cdf output path: {cdf_file}", file=file)
            print(f"Cdh output path: {cdh_file}", file=file)
            print(f"Console output path: {console_output_file}", file=file)
            print(f"Cdf with random rate: {self.w_rr}", file=file)
            print(f"Cdf with scatter rate: {self.w_sc}", file=file)
            print(f"Cdf with attenuation correction factor: {self.w_acf}", file=file)
            print(f"Cdf with DOI: {self.w_doi}", file=file)
            print(f"Cdf with TOF: {self.w_tof}", file=file)
            print(f"\n", file=file)
            # print(f"time window setting: {self.tw}*2.5 ns", file=file)
            print(f"time window setting: {self.tw} ps", file=file)
            print(f"Acquisition total time: {self.total_time * 60} s", file=file)
            # print(f"Acquisition total time: {self.total_time * 5} s", file=file)
            # print(f"Acquisition total time: {self.total_time * 6} s", file=file)
            print(f"\n", file=file)

        num_counts = 0
        for time in tqdm(self.time_loop, desc="Stacking coins..."):
            for pack in range(1, 300):
            # for pack in range(1, 2):

                coins_file_path = "%s/coins%d_%d.dat" % (self.file_path, time, pack)
                # coins_file_path = "%s/coins%d.dat" % (self.file_path, time)
                if not os.path.exists(coins_file_path):
                    continue
                coin_info = coin_dat_to_info(file_path=coins_file_path, with_multiple=self.w_multi, eng_window=self.w_engwindow, time_window=self.tw, return_tof=self.w_tof)
                coins_data, doi_data = coin_info[0:2]
                # coin_info = gate_coin_dat_to_info(file_path=coins_file_path, with_multiple=self.w_multi, eng_window=self.w_engwindow, time_window=self.tw, return_tof=self.w_tof)
                # coins_data = coin_info[0]
                if coins_data.max() > self.scanner_option.crystal_num | coins_data.max() < self.scanner_option.crystal_num * 0.5:
                    print("Geometry file might be wrong...")

                if self.w_tof:
                    tof_data = coin_info[2]
                    if tof_data.max() > self.tof_range[1]:
                        self.tof_range[1] = tof_data.max()
                    if tof_data.min() < self.tof_range[0]:
                        self.tof_range[0] = tof_data.min()

                if self.w_rr:
                    random_rate = self.get_randoms_rate(coins_data)
                if self.w_acf:
                    ac_factor = self.get_ac_factor(coins_data)
                if self.w_norm:
                    norm_factor = self.get_norm_factor(coins_data)
                if self.w_sc:
                    sc_factor = self.get_scatter_factor(coins_data)

                # cdf 数据写入
                op_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
                if self.w_doi:
                    # [1:1]会插入元素在原索引为1的元素前面
                    # [1:2]则会替换原索引为1的元素
                    op_dtype[1:1] = [('poi_x_1', 'f4'), ('poi_x_2', 'f4'), ('poi_y_1', 'f4'), ('poi_y_2', 'f4'), ('poi_z_1', 'f4'), ('poi_z_2', 'f4')]
                if self.w_tof:
                    # op_dtype[1:1] = [('tof_resolution', 'f4')]
                    op_dtype[1:1] = [('tof', 'f4')]
                if self.w_norm:
                    op_dtype[1:1] = [('norm', 'f4')]
                if self.w_rr:
                    op_dtype[1:1] = [('random_rate', 'f4')]
                if self.w_sc:
                    op_dtype[1:1] = [('sc_factor', 'f4')]
                if self.w_acf:
                    op_dtype[1:1] = [('acf_factor', 'f4')]

                structured_array = np.empty(coins_data.shape[0], dtype=op_dtype)
                if self.w_tof:
                    # structured_array['tof_resolution'] = np.ones(coins_data.shape[0])
                    structured_array['tof'] = tof_data
                if self.w_doi:
                    structured_array['poi_x_1'] = np.zeros(coins_data.shape[0])
                    structured_array['poi_x_2'] = np.zeros(coins_data.shape[0])
                    structured_array['poi_y_1'] = np.zeros(coins_data.shape[0])
                    structured_array['poi_y_2'] = np.zeros(coins_data.shape[0])
                    structured_array['poi_z_1'] = doi_data[:, 0]
                    structured_array['poi_z_2'] = doi_data[:, 1]
                if self.w_norm:
                    structured_array['norm'] = norm_factor
                if self.w_rr:
                    structured_array['random_rate'] = random_rate
                if self.w_sc:
                    structured_array['sc_factor'] = sc_factor
                if self.w_acf:
                    structured_array['acf_factor'] = ac_factor

                structured_array['time'] = 0
                structured_array['castor_id_1'] = coins_data[:, 0]
                structured_array['castor_id_2'] = coins_data[:, 1]

                buffer = structured_array.tobytes()
                with open(cdf_file, 'ab') as cdf:
                    cdf.write(buffer)

                num_counts += coins_data.shape[0]
                # 控制台文本输出
                with open(console_output_file, 'a') as file:
                    print(f"Current running file: {coins_file_path}", file=file)
                    print(f"\tCoincidence counts: {coins_data.shape[0]}", file=file)
                    print(f"\tTotal count up to now.: {num_counts}", file=file)
                    print(f"\n", file=file)

        with open(console_output_file, 'a') as file:
            print(f"Number of counts: {num_counts}", file=file)
            print(f"\n", file=file)

        # cdh 文件输出
        with open(cdh_file, 'w') as file:
            print(f"Data filename: {self.save_filename}.cdf", file=file)
            print(f"Number of events: {num_counts}", file=file)
            print(f"Data mode: list-mode", file=file)
            print(f"Data type: PET", file=file)
            print(f"Start time (s): 0", file=file)
            # print(f"Duration (s): {self.total_time * 5}", file=file)
            print(f"Duration (s): {self.total_time * 60}", file=file)
            # print(f"Duration (s): {self.total_time * 6}", file=file)
            # print(f"Duration (s): {self.total_time}", file=file)
            print(f"Scanner name: {self.scanner_option.scanner}", file=file)
            print(f"Calibration factor: 1", file=file)
            print(f"lsotope: unknown", file=file)
            print(f"Random correction flag: {int(self.w_rr)}", file=file)
            print(f"Scatter correction flag: {int(self.w_sc)}", file=file)
            if self.w_doi:
                print(f"POI info flag: {int(self.w_doi)}", file=file)
                print(f"POI resolution: {0.1},{0.1},{0.1}", file=file)
            if self.w_tof:
                print(f"TOF information flag: {int(self.w_tof)}", file=file)
                print(f"TOF resolution (ps): {self.tof_resolution}", file=file)
                print(f"Per event TOF resolution flag: {0}", file=file)
                print(f"List TOF measurement range (ps): {int(self.tof_range[1] - self.tof_range[0])}", file=file)
            print(f"Attenuation correction flag: {int(self.w_acf)}", file=file)
            print(f"Normalization correction flag: {int(self.w_norm)}", file=file)
            print(f"\n", file=file)

        # 清理变量
        self.norm_factor = None
        self.ac_factor = None
