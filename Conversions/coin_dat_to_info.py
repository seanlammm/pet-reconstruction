import numpy as np
import struct
import os


# def get_coins_data(file_path):
#     '''
#     Read Coins of 20250917 new data type
#     :param file_path:
#     :return:
#     '''
#     with open(file_path, 'rb') as f:
#         raw_data = f.read()
#     block_size = (1*0+2*1+4*3+8*1) # 20250917 new 32-bit tof
#     data_len = int(os.path.getsize(file_path) / block_size)
#     if os.path.getsize(file_path) / block_size % 1 != 0:
#         raise Exception("Error in size.")
#
#     format_uint8 = f'{data_len}B'  # N 个 uint8
#     format_uint16 = f'{data_len}H'  # N 个 uint16
#     format_uint32 = f'{data_len}I'  # N 个 uint32
#     format_float = f'{data_len}f'  # N 个 float
#     format_uint64 = f'{data_len}Q'  # N 个 uint64
#
#     def get_offset(uint8_num, uint16_num, uint32_num, float_num):
#         uint8_offset = data_len
#         uint16_offset = data_len * 2
#         uint32_offset = data_len * 4
#         float_offset = data_len * 4
#
#         offset = uint8_offset * uint8_num + uint16_offset * uint16_num + uint32_offset * uint32_num + float_offset * float_num
#         return offset
#
#     castor_id = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(0, 0, 0, 0))
#     doi = np.frombuffer(raw_data, dtype=np.float32, count=data_len, offset=get_offset(0, 0, 1, 0))
#     energy = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 0,1, 1))
#     multiple_flag = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(0, 1, 1, 1))
#
#     time = np.frombuffer(raw_data, dtype=np.uint64, count=data_len, offset=get_offset(0, 1, 2, 1))
#
#     return castor_id, doi, energy, multiple_flag, time

# def get_coins_data(file_path):
#     '''
#     Read coins of old 22bit data type
#     :param file_path:
#     :return:
#     '''
#     with open(file_path, 'rb') as f:
#         raw_data = f.read()
#     block_size = (1*3+2*3+4*4)  # 22-bit
#     data_len = int(os.path.getsize(file_path) / block_size)
#     if os.path.getsize(file_path) / block_size % 1 != 0:
#         raise Exception("Error in size.")
#
#     format_uint8 = f'{data_len}B'  # N 个 uint8
#     format_uint16 = f'{data_len}H'  # N 个 uint16
#     format_uint32 = f'{data_len}I'  # N 个 uint32
#     format_float = f'{data_len}f'  # N 个 float
#     format_uint64 = f'{data_len}Q'  # N 个 uint64
#
#     def get_offset(uint8_num, uint16_num, uint32_num, float_num):
#         uint8_offset = data_len
#         uint16_offset = data_len * 2
#         uint32_offset = data_len * 4
#         float_offset = data_len * 4
#
#         offset = uint8_offset * uint8_num + uint16_offset * uint16_num + uint32_offset * uint32_num + float_offset * float_num
#         return offset
#
#     y_pos = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 0, 0, 0))
#     x_pos = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 1, 0, 0))
#     crystal_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(0, 2, 0, 0))
#     castor_id = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(1, 2, 0, 0))
#     doi = np.frombuffer(raw_data, dtype=np.float32, count=data_len, offset=get_offset(1, 2, 1, 0))
#     cir_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(1, 2, 1, 1))
#     axi_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(2, 2, 1, 1))
#     energy = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(3, 2, 1, 1))
#     multiple_flag = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(3, 3, 1, 1))
#
#     time = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(3, 3, 2, 1))
#
#     return castor_id, doi, energy, multiple_flag, time


def get_coins_data(file_path):
    '''
    Read coins of old 32bit data(before 20250917)
    :param file_path:
    :return:
    '''
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    block_size = (1 * 3 + 2 * 3 + 4 * 3 + 8 * 1)  # 32-bit tof
    data_len = int(os.path.getsize(file_path) / block_size)
    if os.path.getsize(file_path) / block_size % 1 != 0:
        raise Exception("Error in size.")

    format_uint8 = f'{data_len}B'  # N 个 uint8
    format_uint16 = f'{data_len}H'  # N 个 uint16
    format_uint32 = f'{data_len}I'  # N 个 uint32
    format_float = f'{data_len}f'  # N 个 float
    format_uint64 = f'{data_len}Q'  # N 个 uint64

    def get_offset(uint8_num, uint16_num, uint32_num, float_num):
        uint8_offset = data_len
        uint16_offset = data_len * 2
        uint32_offset = data_len * 4
        float_offset = data_len * 4

        offset = uint8_offset * uint8_num + uint16_offset * uint16_num + uint32_offset * uint32_num + float_offset * float_num
        return offset

    y_pos = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 0, 0, 0))
    x_pos = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(0, 1, 0, 0))
    crystal_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(0, 2, 0, 0))
    castor_id = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(1, 2, 0, 0))
    doi = np.frombuffer(raw_data, dtype=np.float32, count=data_len, offset=get_offset(1, 2, 1, 0))
    cir_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(1, 2, 1, 1))
    axi_id = np.frombuffer(raw_data, dtype=np.uint8, count=data_len, offset=get_offset(2, 2, 1, 1))
    energy = np.frombuffer(raw_data, dtype=np.uint16, count=data_len, offset=get_offset(3, 2, 1, 1))
    multiple_flag = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(3, 3, 1, 1))

    time = np.frombuffer(raw_data, dtype=np.uint64, count=data_len, offset=get_offset(3, 3, 2, 1))

    return castor_id, doi, energy, multiple_flag, time


def get_gate_coins_data(file_path):
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

    return castor_id, energy, multiple_flag, time


def get_delayed_coins_data(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    # block_size = (2 * 1 + 4 * 2 + 8 * 1)  # gate coins
    block_size = (2 * 1 + 4 * 3)  # gate coins
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
    time = np.frombuffer(raw_data, dtype=np.uint32, count=data_len, offset=get_offset(0, 1, 2, 0, 0))

    return castor_id, energy, multiple_flag, time


def coin_dat_to_info(file_path, with_multiple, eng_window, time_window, return_tof=False):
    castor_id, doi, energy, multiple_flag, time = get_coins_data(file_path)
    time = (time - np.min(time)).astype(np.float64)
    # castor_id = castor_id.copy()
    # castor_id[castor_id >= 57600] -= 57600

    if not with_multiple:
        castor_id = castor_id[multiple_flag == 0]
        doi = doi[multiple_flag == 0]
        energy = energy[multiple_flag == 0]
        time = time[multiple_flag == 0]

    data_len = int(castor_id.shape[0] / 2)
    castor_info = np.column_stack((castor_id[:data_len], castor_id[data_len:]))
    energy_info = np.column_stack((energy[:data_len], energy[data_len:]))
    time_info = np.column_stack((time[:data_len], time[data_len:]))
    doi_info = np.column_stack((doi[:data_len], doi[data_len:]))

    # # with only first coins in multiple coins
    # first_multi_coins = np.where(np.diff(multiple_flag[:data_len]) != 0)[0] + 1  # include first 0 events
    # valid_index = np.zeros(castor_info.shape[0], dtype=bool)
    # valid_index[multiple_flag[:data_len] == 0] = 1
    # valid_index[first_multi_coins] = 1
    # castor_info = castor_info[valid_index, :]
    # time_info = time_info[valid_index, :]
    # doi_info = doi_info[valid_index, :]
    # energy_info = energy_info[valid_index, :]

    if eng_window is not None:  # 加能窗
        in_flag = (eng_window[0] <= energy_info[:, 0]) & (energy_info[:, 0] <= eng_window[1]) & (
                    eng_window[0] <= energy_info[:, 1]) & (energy_info[:, 1] <= eng_window[1])
        castor_info = castor_info[in_flag, :]
        time_info = time_info[in_flag, :]
        doi_info = doi_info[in_flag, :]

    tof_info = (time_info[:, 0] - time_info[:, 1]).astype(np.float32)
    in_time_window = np.abs(tof_info) <= time_window
    castor_info = castor_info[in_time_window, :]
    doi_info = doi_info[in_time_window, :]
    tof_info = tof_info[in_time_window]

    func_return = [castor_info, doi_info]
    if return_tof:
        func_return.append(tof_info)
    return func_return


def gate_coin_dat_to_info(file_path, with_multiple, eng_window, time_window, return_tof=False):
    castor_id, energy, multiple_flag, time = get_gate_coins_data(file_path)
    time = (time - np.min(time)).astype(np.float64)

    if not with_multiple:
        castor_id = castor_id[multiple_flag == 0]
        energy = energy[multiple_flag == 0]
        time = time[multiple_flag == 0]

    data_len = int(castor_id.shape[0] / 2)
    castor_info = np.column_stack((castor_id[:data_len], castor_id[data_len:]))
    energy_info = np.column_stack((energy[:data_len], energy[data_len:]))
    time_info = np.column_stack((time[:data_len], time[data_len:]))

    if eng_window is not None:  # 加能窗
        in_flag = (eng_window[0] <= energy_info[:, 0]) & (energy_info[:, 0] <= eng_window[1]) & (
                    eng_window[0] <= energy_info[:, 1]) & (energy_info[:, 1] <= eng_window[1])
        castor_info = castor_info[in_flag, :]
        time_info = time_info[in_flag, :]

    tof_info = (time_info[:, 0] - time_info[:, 1]).astype(np.float32)
    # in_time_window = np.abs(tof_info) <= time_window
    # castor_info = castor_info[in_time_window, :]

    func_return = [castor_info]
    if return_tof:
        func_return.append(tof_info)
    return func_return


def delayed_coin_dat_to_info(file_path, with_multiple, tw, eng_window, if_unique=True):
    castor_id, energy, multiple_flag, time = get_delayed_coins_data(file_path)
    time = (time - np.min(time)).astype(np.float64)

    if not with_multiple:
        castor_id = castor_id[multiple_flag == 0]
        energy = energy[multiple_flag == 0]
        time = time[multiple_flag == 0]

    data_len = int(castor_id.shape[0] / 2)
    castor_info = np.column_stack((castor_id[:data_len], castor_id[data_len:]))
    energy_info = np.column_stack((energy[:data_len], energy[data_len:]))
    time_info = np.column_stack((time[:data_len], time[data_len:]))

    if eng_window is not None:  # 加能窗
        in_flag = (eng_window[0] <= energy_info[:, 0]) & (energy_info[:, 0] <= eng_window[1]) & (
                eng_window[0] <= energy_info[:, 1]) & (energy_info[:, 1] <= eng_window[1])
        castor_info = castor_info[in_flag, :]
        time_info = time_info[in_flag, :]

    tof_info = (time_info[:, 0] - time_info[:, 1]).astype(np.float32)
    # in_time_window = np.abs(tof_info) <= tw
    # castor_info = castor_info[in_time_window, :]

    if if_unique:
        uni_ids, counts = np.unique(castor_info, axis=0, return_counts=True)
        return uni_ids, counts
    else:
        return castor_info

