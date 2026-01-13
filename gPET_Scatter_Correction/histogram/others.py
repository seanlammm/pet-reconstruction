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


def cdf_to_block_base_sinogram(cdf_path: str, tof_option: TOFOption, wtof=1, wrr=0, scanner_option=0, num_of_chunk=30):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if wtof:
        # op_dtype[1:1] = [('tof_resolution', 'f4')]
        ip_type[1:1] = [('tof', 'f4')]
    if wrr:
        ip_type[1:1] = [('random_rate', 'f4')]

    input_cdf = np.fromfile(cdf_path, dtype=ip_type)
    michelogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins], dtype=np.float32)
    for chunk in tqdm(range(num_of_chunk), desc="Processing chunks..."):
        chunk_size = int(input_cdf.shape[0] // num_of_chunk)
        start_index = chunk_size * chunk
        end_index = chunk_size * (chunk + 1) if chunk != num_of_chunk - 1 else input_cdf.shape[0]

        castor_id_1 = input_cdf["castor_id_1"][start_index:end_index]
        castor_id_2 = input_cdf["castor_id_2"][start_index:end_index]
        tof = input_cdf["tof"][start_index:end_index]

        swap_flag = castor_id_1 > castor_id_2
        castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_2[swap_flag].copy(), castor_id_1[swap_flag].copy(), -tof[swap_flag].copy()
        tof_bin_index, rm_option = tof_option.get_tof_bin_index(tof)

        castor_id_1, castor_id_2, tof_bin_index = castor_id_1[~rm_option], castor_id_2[~rm_option], tof_bin_index[~rm_option]
        miche_index, miche_rm_option = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
        tof_bin_index = tof_bin_index[~miche_rm_option]
        index = tuple(np.column_stack((tof_bin_index, miche_index)).astype(int).T)
        np.add.at(michelogram, index, 1)

    return michelogram


def cdf_to_crystal_base_sinogram(cdf_path: str, wtof, wrr, scanner_option, tof_option, num_of_chunk=30):
    ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if wtof:
        # op_dtype[1:1] = [('tof_resolution', 'f4')]
        ip_type[1:1] = [('tof', 'f4')]
    if wrr:
        ip_type[1:1] = [('random_rate', 'f4')]

    input_cdf = np.fromfile(cdf_path, dtype=ip_type)
    michelogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins], dtype=np.float32)
    for chunk in tqdm(range(num_of_chunk), desc="Processing chunks..."):
        chunk_size = int(input_cdf.shape[0] // num_of_chunk)
        start_index = chunk_size * chunk
        end_index = chunk_size * (chunk + 1) if chunk != num_of_chunk - 1 else input_cdf.shape[0]

        castor_id_1 = input_cdf["castor_id_1"][start_index:end_index]
        castor_id_2 = input_cdf["castor_id_2"][start_index:end_index]
        tof = input_cdf["tof"][start_index:end_index]

        swap_flag = castor_id_1 > castor_id_2
        castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_2[swap_flag].copy(), castor_id_1[swap_flag].copy(), -tof[swap_flag].copy()

        tof_bin_index, rm_option = tof_option.get_tof_bin_index(tof)
        castor_id_1, castor_id_2, tof_bin_index = castor_id_1[~rm_option], castor_id_2[~rm_option], tof_bin_index[~rm_option]
        miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option, np.column_stack((castor_id_1, castor_id_2)), False)
        tof_bin_index = tof_bin_index[~miche_rm_option]
        index = tuple(np.column_stack((tof_bin_index, miche_index)).astype(int).T)
        np.add.at(michelogram, index, 1)
    return michelogram


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
