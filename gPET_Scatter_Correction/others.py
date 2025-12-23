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
    sigma = tof_resolution / (2 * np.sqrt(2 * np.log(2)))  # ≈ fwhm / 2.3548
    gaussian_rands = np.random.normal(loc=0, scale=sigma, size=tof.shape[0])
    tof += gaussian_rands
    return tof


def cdf_to_block_base_sinogram(cdf_path: str, tof_option: TOFOption, with_blur=0, wtof=1, wrr=0, scanner_option=0, num_of_chunk=30):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if wtof:
        # op_dtype[1:1] = [('tof_resolution', 'f4')]
        ip_type[1:1] = [('tof', 'f4')]
    if wrr:
        ip_type[1:1] = [('random_rate', 'f4')]

    input_cdf = np.fromfile(cdf_path, dtype=ip_type)
    michelogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    for chunk in tqdm(range(num_of_chunk), desc="Processing chunks..."):
        chunk_size = int(input_cdf.shape[0] // num_of_chunk)
        start_index = chunk_size * chunk
        end_index = chunk_size * (chunk + 1) if chunk != num_of_chunk - 1 else input_cdf.shape[0]

        castor_id_1 = input_cdf["castor_id_1"][start_index:end_index]
        castor_id_2 = input_cdf["castor_id_2"][start_index:end_index]
        tof = input_cdf["tof"][start_index:end_index]
        if with_blur:
            tof = tof_blurring(tof, tof_option.tof_resolution)

        swap_flag = tof > 0
        castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_1[swap_flag].copy(), castor_id_2[swap_flag].copy(), -tof[swap_flag].copy()
        tof_bin_index, rm_option = tof_option.get_tof_bin_index(tof)
        miche_index, miche_rm_option = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
        tof_bin_index = tof_bin_index[~miche_rm_option]
        index = tuple(np.column_stack((tof_bin_index, miche_index)).astype(int).T)
        np.add.at(michelogram, index, 1)

    return michelogram


def cdf_to_crystal_base_sinogram_to_block_base_sinogram(cdf_path, tof_bin_num, with_blur=0):
    _,  castor_ids_in_unique, crystal_base_michelogram_counts = cdf_to_crystal_base_sinogram(cdf_path, tof_bin_num, with_blur=with_blur, w_tof=1)

    os.chdir(r"/")
    scanner_option = ScannerOption("PET_11panel_Module_Base")

    miche_index, miche_rm_option = crystal_base_id_to_block_base_sinogram(castor_ids_in_unique[:, 1:])
    castor_ids_in_unique = castor_ids_in_unique[~miche_rm_option, :]

    index = tuple(np.column_stack((castor_ids_in_unique[:, 0], miche_index)).astype(int).T)
    michelogram = np.zeros([tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    np.add.at(michelogram, index, crystal_base_michelogram_counts)
    return michelogram


def cdf_to_crystal_base_sinogram(cdf_path: str, tof_bin_num, with_blur=0, w_tof=1, scanner_option=0, num_of_chunk=30):
    os.chdir(r"/")
    scanner_option = ScannerOption("PET_11panel_LD")
    if w_tof:
        ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    else:
        ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    input_cdf = np.fromfile(cdf_path, dtype=ip_type)
    michelogram = np.zeros([tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    for chunk in tqdm(range(num_of_chunk), desc="Processing chunks..."):
        chunk_size = int(input_cdf.shape[0] // num_of_chunk)
        start_index = chunk_size * chunk
        end_index = chunk_size * (chunk + 1) if chunk != num_of_chunk - 1 else input_cdf.shape[0]

        castor_id_1 = input_cdf["castor_id_1"][start_index:end_index]
        castor_id_2 = input_cdf["castor_id_2"][start_index:end_index]
        tof = input_cdf["tof"][start_index:end_index]

        if w_tof:
            tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
            positive_tof_flag = tof > 0
            temp = castor_id_1[positive_tof_flag].copy()
            castor_id_1[positive_tof_flag] = castor_id_2[positive_tof_flag]
            castor_id_2[positive_tof_flag] = temp
            tof[positive_tof_flag] *= -1

            if with_blur:
                tof = tof_blurring(tof, 300)

            tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
            tof_bin_index[tof_bin_index < 0] = 0
            tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1

        unique, counts = np.unique(tof_bin_index.astype(int)*scanner_option.crystal_per_layer**2 + castor_id_1.astype(int)*scanner_option.crystal_per_layer + castor_id_2.astype(int), return_counts=True)
        unique_ids = np.column_stack((
            unique // scanner_option.crystal_per_layer**2,
            (unique % scanner_option.crystal_per_layer**2) // scanner_option.crystal_per_layer,
            unique % scanner_option.crystal_per_layer
        ))  # [tof, castor_id_1, castor_id_2]
        miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option, unique_ids[:, 1:], False)

        unique_ids = unique_ids[~miche_rm_option, :]
        counts = counts[~miche_rm_option]
        index = tuple(np.column_stack((unique_ids[:, 0], miche_index)).astype(int).T)
        np.add.at(michelogram, index, counts)

        # michelogram = scipy.ndimage.gaussian_filter(michelogram, sigma=3, order=0, mode='reflect', truncate=4.0, axes=(-2, -1))
        michelogram_counts = michelogram[index]

    return michelogram, unique_ids, michelogram_counts
