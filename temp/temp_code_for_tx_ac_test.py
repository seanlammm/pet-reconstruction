import numpy as np
import os
from Conversions.castor_id_to_sinogram import get_sinogram, check_id_in_sinogram


def read_cdf(cdf_path, start_index, end_index=None):
    cdf_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    block_size = np.empty(0, dtype=cdf_dtype).itemsize
    file_size = os.path.getsize(cdf_path) / block_size
    if end_index is None or end_index > file_size:
        end_index = file_size

    start_index = int(start_index)
    end_index = int(end_index)
    with open(cdf_path, 'rb') as f:
        f.seek(block_size * start_index)
        data = f.read(block_size * (end_index - start_index))

    coins = np.frombuffer(data, dtype=cdf_dtype)
    coins = np.column_stack((coins["castor_id_1"], coins["castor_id_2"]))
    return coins


def transform_iq_phantom():
    import numpy as np
    import matplotlib.pyplot as plt

    pet_img = np.fromfile(r"C:\Users\ct-guys\Downloads\DRO_20130529\DRO_20130529\20130528_DRO_DICOM_male\DRO_PET_20130528.raw", dtype=np.float32).reshape([110, 256, 256])

    test = pet_img.copy()
    test[test == 0] = np.nan  # air
    test[test >= 10000] = np.nan  # sphere inside
    test[(test >= 3200) & (test < 3650)] = np.nan  # 背景 water
    test[(test >= 0) & (test < 400)] = np.nan  # 右下角的马赛克格子
    plt.imshow(test[40, :, :])
    plt.show(block=True)







