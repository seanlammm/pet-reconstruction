import numpy as np
import struct
import os


def coin_dat_to_only_castor_id(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        data_len = int(os.path.getsize(file_path) / ((2*1+4*4)*2))
        if os.path.getsize(file_path) / (2*1+4*4) % 1 != 0:
            print("error in size.")
        format_uint16 = f'{data_len}H'  # N 个 uint16
        format_uint32 = f'{data_len}I'  # N 个 uint32
        format_float = f'{data_len}f'  # N 个 float

        sum_data = np.zeros([data_len, 2])

        f.seek(0)  # 重置指针
        sum_data[:, 0] = struct.unpack(format_uint32, f.read(data_len * 4))  # castor_id_0
        sum_data[:, 1] = struct.unpack(format_uint32, f.read(data_len * 4))  # castor_id_1

    return sum_data.astype(int)