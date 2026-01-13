import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram
from Generals.TOFGenerals import TOFOption


def crystal_base_id_to_block_base_sinogram(castor_ids):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    axi_ids = castor_ids // 480 // 10
    cir_ids = castor_ids % 480 // 8
    mb_castor_ids = axi_ids * 60 + cir_ids

    miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option, mb_castor_ids, get_ssrb=False)
    return miche_index, miche_rm_option


def tof_blurring(tof, tof_resolution):
    np.random.seed(42)
    sigma = tof_resolution / (2 * np.sqrt(2 * np.log(2)))
    gaussian_rands = np.random.normal(loc=0, scale=sigma, size=tof.shape[0])
    blurred_tof = tof + gaussian_rands
    return blurred_tof


def castor_id_to_block_base_sinogram(cdf_data, scanner_option: ScannerOption, tof_option: TOFOption):
    michelogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins], dtype=np.float32)

    castor_ids = cdf_data[:, -2:]
    axi_ids = castor_ids // 480 // 10
    cir_ids = castor_ids % 480 // 8
    mb_castor_ids = axi_ids * 60 + cir_ids

    tof = cdf_data[:, 2]
    flag = mb_castor_ids[:, 0] > mb_castor_ids[:, 1]
    mb_castor_ids[flag, 0], mb_castor_ids[flag, 1], tof[flag] = mb_castor_ids[flag, 1].copy(), mb_castor_ids[flag, 0].copy(), -tof[flag].copy()
    tof_bin_index, _ = tof_option.get_tof_bin_index(tof)

    # tof_bin_index = cdf_data[:, 3]
    sinogram_index, rm_option = castor_id_to_sinogram(scanner_option=scanner_option, castor_ids=mb_castor_ids, get_ssrb=False)
    index = tuple(np.column_stack((tof_bin_index[~rm_option], sinogram_index)).astype(int).T)
    np.add.at(michelogram, index, 1)
    return michelogram


def get_block_base_sinogram_index(cdf_data, scanner_option: ScannerOption, tof_option: TOFOption):
    castor_ids = cdf_data[:, -2:]
    axi_ids = castor_ids // 480 // 10
    cir_ids = castor_ids % 480 // 8
    mb_castor_ids = axi_ids * 60 + cir_ids

    tof = cdf_data[:, 2]
    flag = mb_castor_ids[:, 0] > mb_castor_ids[:, 1]
    mb_castor_ids[flag, 0], mb_castor_ids[flag, 1], tof[flag] = mb_castor_ids[flag, 1].copy(), mb_castor_ids[flag, 0].copy(), -tof[flag].copy()
    tof_bin_index, _ = tof_option.get_tof_bin_index(tof)

    # tof_bin_index = cdf_data[:, 3]
    sinogram_index, rm_option = castor_id_to_sinogram(scanner_option=scanner_option, castor_ids=mb_castor_ids, get_ssrb=False)
    index = tuple(np.column_stack((tof_bin_index[~rm_option], sinogram_index)).astype(int).T)
    return index, rm_option


def get_cdf_info(cdf_path, tof_option: TOFOption, wrr, wtof):

    ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if wtof:
        ip_type[1:1] = [('tof', 'f4')]
    if wrr:
        ip_type[1:1] = [('random_rate', 'f4')]

    cdf = np.fromfile(cdf_path, dtype=ip_type)
    time = cdf["time"].astype(float)
    castor_id_1 = cdf["castor_id_1"].astype(float)
    castor_id_2 = cdf["castor_id_2"].astype(float)
    rr = cdf["random_rate"].astype(float) if wrr else np.zeros_like(time, dtype=float)
    tof = cdf["tof"].astype(float)

    swap_flag = castor_id_1 > castor_id_2
    castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_2[swap_flag].copy(), castor_id_1[swap_flag].copy(), -tof[swap_flag].copy()

    tof_bin_index, tof_rm_option = tof_option.get_tof_bin_index(tof)
    target_cdf = np.column_stack((time, rr, tof, tof_bin_index, castor_id_1, castor_id_2))
    target_cdf = target_cdf[~tof_rm_option, :]

    return target_cdf
