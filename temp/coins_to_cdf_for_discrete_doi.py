import numpy as np
import struct
import os
from tqdm import tqdm
from Generals.ScannerGenerals import ScannerOption
from Conversions.coin_dat_to_info import coin_dat_to_info, coin_dat_to_info_wdoi, coin_dat_to_info_discrete_multi_layer


class CoinsToCdf:
    def __init__(self, scanner_option: ScannerOption, file_path, save_path, save_filename, total_time, time_window, with_random_rate: bool = False, with_attenuation_correction_factor: bool = False, with_normalization_factor: bool = True, with_scatter_correction_factor: bool = False, with_doi: bool = False, ac_factor_file_path=None, norm_factor_file_path=None, sc_factor_file_path=None, with_multiple_coins=False, with_energy_window=None):
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
        self.path_to_acf = ac_factor_file_path
        self.path_to_normf = norm_factor_file_path
        self.path_to_scf = sc_factor_file_path
        self.w_multi = with_multiple_coins
        self.w_engwindow = with_energy_window
        self.w_doi = with_doi

        if self.total_time.shape[0] == 1:
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

    def get_randoms_rate(self, coins_data=None):
        if coins_data is None:
            sum_singles_counts = np.zeros([self.scanner_option.crystal_per_layer])
            for time in tqdm(self.time_loop, desc="Collecting singles counts"):
                for pack in range(1, 300):
                    singles_counts_file = os.path.join(self.file_path, "singles_counts%d_%d.dat" % (time, pack))
                    if not os.path.exists(singles_counts_file) and pack==1:
                        print("file: %s" % singles_counts_file)
                        raise Exception("文件不存在.")
                    if not os.path.exists(singles_counts_file):
                        break

                    item_size = np.dtype(np.uint16).itemsize
                    start_idx = int(self.w_engwindow[0] // 10)
                    end_idx = int(self.w_engwindow[1] // 10)
                    counts = (end_idx - start_idx) * 57600  # +1 代表包含右边界
                    offset = start_idx * 57600 * item_size
                    singles_counts = np.fromfile(singles_counts_file, dtype=np.uint16, count=counts, offset=offset).reshape([-1, 57600])
                    sum_singles_counts += np.sum(singles_counts, axis=0)

            total_time = self.total_time * 60  # unit: second
            sum_singles_rates = sum_singles_counts / total_time
            return sum_singles_rates

        else:
            coin_time_window = self.tw * 2.5 * 1e-9  # time_window * seconds per time window
            singles_rate_0 = self.sum_singles_rates[coins_data[:, 0]]
            singles_rate_1 = self.sum_singles_rates[coins_data[:, 1]]
            randoms_rate = 2 * coin_time_window * singles_rate_0 * singles_rate_1

            return randoms_rate.astype(np.float32)

    def run(self):
        if self.w_rr:
            self.save_filename += "_wrr"
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
            raise Exception("需要手动删除已存在的输出文件")

        with open(console_output_file, 'a') as file:
            print(f"Cdf output path: {cdf_file}", file=file)
            print(f"Cdh output path: {cdh_file}", file=file)
            print(f"Console output path: {console_output_file}", file=file)
            print(f"Cdf with random rate: {self.w_rr}", file=file)
            print(f"Cdf with scatter rate: {self.w_sc}", file=file)
            print(f"Cdf with attenuation correction factor: {self.w_acf}", file=file)
            print(f"\n", file=file)
            print(f"time window setting: {self.tw}*2.5 ns", file=file)
            print(f"Acquisition total time: {self.total_time * 60} s", file=file)
            # print(f"Acquisition total time: {self.total_time} s", file=file)
            print(f"\n", file=file)

        num_counts = 0

        # 20250416_t3
        error_file = [[5, 21], [7, 24], [8, 24], [8, 26],
                      [10, 31], [13, 23], [13, 29], [13, 38],
                      [14, 9], [14, 15], [14, 8], [14, 25],
                      [14, 2]]

        # # 20250416_t11
        # error_file = [[1, 5], [1, 15], [2, 6], [2, 17], [2, 18], [3, 12],
        #               [3, 17], [4, 2], [4, 8], [4, 15], [5, 5], [5, 12],
        #               [5, 11], [6, 6], [6, 8], [6, 11], [6, 14], [7, 1],
        #               [7, 13], [7, 6], [7, 9], [7, 17], [8, 1], [8, 2],
        #               [8, 16], [8, 11], [8, 12], [8, 8], [9, 1], [9, 12],
        #               [9, 9], [10, 4], [11, 12], [11, 6], [11, 16], [12, 10],
        #               [12, 8], [12, 4], [13, 10], [13, 11], [13, 6], [14, 5],
        #               [14, 6], [15, 1], [15, 6], [16, 2], [16, 7], [16, 3],
        #               [16, 9], [17, 9], [17, 8], [18, 7], [18, 8], [18, 6],
        #               [19, 7], [19, 8], [19, 2], [19, 12], [20, 12], [20, 3],
        #               [20, 4], [20, 5], [20, 7]]
        # error_file = np.array(error_file, dtype=int)

        for time in tqdm(self.time_loop, desc="Stacking coins..."):
            for pack in range(1, 300):
                # 跳过时间锯齿周期错误的数据
                if np.any((error_file[:, 0] == time) & (error_file[:, 1] == pack)):
                    continue

                coins_file_path = "%s/coins%d_%d.dat" % (self.file_path, time, pack)
                if not os.path.exists(coins_file_path):
                    continue
                coins_data = coin_dat_to_info_discrete_multi_layer(coins_file_path, with_multiple=self.w_multi, tw=self.tw, eng_window=self.w_engwindow)

                if self.w_rr:
                    random_rate = self.get_randoms_rate(coins_data)

                # cdf 数据写入
                op_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
                if self.w_rr:
                    op_dtype[1:1] = [('random_rate', 'f4')]

                structured_array = np.empty(coins_data.shape[0], dtype=op_dtype)
                if self.w_rr:
                    structured_array['random_rate'] = random_rate
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
        cdf_name = cdf_file.split("\\")[-1]
        with open(cdh_file, 'w') as file:
            print(f"Data filename: {cdf_name}", file=file)
            print(f"Number of events: {num_counts}", file=file)
            print(f"Data mode: list-mode", file=file)
            print(f"Data type: PET", file=file)
            print(f"Start time (s): 0", file=file)
            print(f"Duration (s): {self.total_time * 60}", file=file)
            # print(f"Duration (s): {self.total_time}", file=file)
            print(f"Scanner name: {self.scanner_option.scanner}", file=file)
            print(f"Calibration factor: 1", file=file)
            print(f"lsotope: unknown", file=file)
            print(f"Random correction flag: {int(self.w_rr)}", file=file)
            print(f"Scatter correction flag: {int(self.w_sc)}", file=file)
            if self.w_doi:
                print(f"POI info flag: {int(self.w_doi)}", file=file)
                print(f"POI resolution: {0.1},{0.1},{0.1}", file=file)
            print(f"Attenuation correction flag: {int(self.w_acf)}", file=file)
            print(f"Normalization correction flag: {int(self.w_norm)}", file=file)
            print(f"\n", file=file)


if __name__ == "__main__":
    os.chdir(r"D:\linyuejie\git-project\pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD_2layers")

    coins2cdf = CoinsToCdf(
        scanner_option=scanner_option,
        file_path=r"E:\acquisition\20250416\acq_data_t3\coins",
        save_path=r"E:\acquisition\20250416\acq_data_t3\cdf",
        save_filename="t3_rm_error_data_discrete_double_layer",
        total_time=20,
        time_window=3,
        with_random_rate=True,
        with_attenuation_correction_factor=False,
        with_normalization_factor=False,
        with_scatter_correction_factor=False,
        with_doi=False,
        sc_factor_file_path="",
        with_multiple_coins=True,
        with_energy_window=[411, 611]
    )
    coins2cdf.run()
