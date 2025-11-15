import numpy as np
from tqdm import tqdm
import os
from Generals.ScannerGenerals import ScannerOption
import scipy.io as scio
import matplotlib.pyplot as plt


def get_singles_rate(scanner_option: ScannerOption, input_file_dir, output_file_dir, eng_window, total_time):
    eng_window = eng_window // 10 * 10

    num_of_crystal = scanner_option.crystal_num
    sum_singles = np.zeros([num_of_crystal], dtype=np.float64)
    for t in tqdm(range(1, 50)):
        for p in range(1, 150):
            singles_file = os.path.join(input_file_dir, "singles_counts%d_%d.dat" % (t, p))
            if not os.path.exists(singles_file):
                continue

            item_size = np.dtype(np.uint16).itemsize
            start_idx = int(eng_window[0] // 10)
            end_idx = int(eng_window[1] // 10)
            counts = (end_idx - start_idx) * num_of_crystal  # +1 代表包含右边界
            offset = start_idx * num_of_crystal * item_size
            singles_counts = np.fromfile(singles_file, dtype=np.uint16, count=counts, offset=offset).reshape([-1, num_of_crystal])
            sum_singles += np.sum(singles_counts, axis=0)
    sum_singles /= (total_time * 60)
    sum_singles.tofile(os.path.join(output_file_dir, "singles_rate_wengwin_%d_%d.raw" % (eng_window[0], eng_window[1])))


def read_delayed_coins(file_path, eng_win, time_window):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    block_size = (2 * 1 + 4 * 2 + 8 * 1)  # gate coins
    data_len = int(os.path.getsize(file_path) / block_size)
    if os.path.getsize(file_path) / block_size % 1 != 0:
        raise Exception("Error in size.")

    def get_offset(uint8_num, uint16_num, uint32_num, float_num, uint64_num):
        uint8_offset = data_len
        uint16_offset = data_len * 2
        uint32_offset = data_len * 4
        float_offset = data_len * 4
        uint64_offset = data_len * 8

        offset = uint8_offset * uint8_num + uint16_offset * uint16_num + uint32_offset * uint32_num + float_offset * float_num + uint64_offset * uint64_num
        return offset

    castor_id = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(0, 0, 0, 0, 0))
    energy = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 0, 1, 0, 0))
    multiple_flag = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(0, 1, 1, 0, 0))
    time = np.frombuffer(raw_data, dtype=np.uint64, count=data_len, offset=get_offset(0, 1, 2, 0, 0))

    data_len = int(castor_id.shape[0] / 2)
    castor_info = np.column_stack((castor_id[:data_len], castor_id[data_len:]))
    energy_info = np.column_stack((energy[:data_len], energy[data_len:]))
    time_info = np.column_stack((time[:data_len], time[data_len:]))

    if eng_window is not None:  # 加能窗
        in_flag = (eng_win[0] <= energy_info[:, 0]) & (energy_info[:, 0] <= eng_win[1]) & (eng_win[0] <= energy_info[:, 1]) & (energy_info[:, 1] <= eng_win[1])
        castor_info = castor_info[in_flag, :]
        time_info = time_info[in_flag, :]

    tof_info = time_info[:, 0] - time_info[:, 1]
    in_time_window = np.abs(tof_info) <= time_window
    castor_info = castor_info[in_time_window, :]
    tof_info = tof_info[in_time_window]
    return castor_info, tof_info


def get_delayed_coins_cdf(scanner_option: ScannerOption, input_file_dir, output_file_dir, eng_window, total_time, time_window):
    """
    :param scanner_option:
    :param input_file_dir:
    :param output_file_dir:
    :param eng_window: [low_cut, high_cut]
    :param total_time: duration time in minutes
    :return: no return, save cdf and cdh in file_dir
    :return:
    """
    eng_window = eng_window // 10 * 10
    output_file_name = "delayed_wtof_wengwin_%d_%d" % (eng_window[0], eng_window[1])
    cdf_file = os.path.join(output_file_dir, "%s.cdf" % output_file_name)
    cdh_path = os.path.join(output_file_dir, "%s.cdh" % output_file_name)

    tof_range = np.zeros(2)
    num_counts = 0
    for time in tqdm(range(50), desc="Stacking coins..."):
        for pack in range(1, 150):
            coins_file_path = "%s/delayed_coins%d_%d.dat" % (input_file_dir, time, pack)
            if not os.path.exists(coins_file_path):
                continue
            id_info, tof_info = read_delayed_coins(file_path=coins_file_path, eng_win=eng_window, time_window=time_window)

            op_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
            structured_array = np.empty(id_info.shape[0], dtype=op_dtype)
            structured_array['time'] = 0
            structured_array['castor_id_1'] = id_info[:, 0]
            structured_array['castor_id_2'] = id_info[:, 1]
            structured_array['tof'] = tof_info

            if tof_info.min() < tof_range[0]:
                tof_range[0] = tof_info.min()
            if tof_info.max() > tof_range[1]:
                tof_range[1] = tof_info.max()

            buffer = structured_array.tobytes()
            with open(cdf_file, 'ab') as cdf:
                cdf.write(buffer)
            num_counts += id_info.shape[0]

    with open(cdh_path, 'w') as file:
        print(f"Data filename: {output_file_name}.cdf", file=file)
        print(f"Number of events: {num_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): {total_time * 60}", file=file)
        print(f"Scanner name: {scanner_option.scanner}", file=file)
        print(f"Calibration factor: 1", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {int(tof_range[1] - tof_range[0])}", file=file)
        print(f"\n", file=file)


if __name__ == "__main__":
    os.chdir(r"D:\linyuejie\git-project\pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    total_time = 20
    eng_window = np.array([410, 610])
    tw = 1000
    input_file_dir = r"E:\data_11panel\20250819\acq_data_t4\coins_by_ps_wTDCCorrection_tw2000ps"
    output_file_dir = r"E:\data_11panel\20250819\acq_data_t4"

    get_delayed_coins_cdf(scanner_option, input_file_dir, output_file_dir, eng_window, total_time, tw)
    # get_singles_rate(scanner_option, input_file_dir, output_file_dir, eng_window, total_time)

    input_file_dir = r"E:\data_11panel\20250827\acq_data_t6\coins"
    output_file_dir = r"E:\data_11panel\20250827\acq_data_t6"
    get_delayed_coins_cdf(scanner_option, input_file_dir, output_file_dir, eng_window, total_time, tw)
    # get_singles_rate(scanner_option, input_file_dir, output_file_dir, eng_window, total_time)
