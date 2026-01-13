import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Generals.TOFGenerals import TOFOption
from gPET_Scatter_Correction.histogram.others import crystal_base_id_to_block_base_sinogram
from Generals.Projector import Projector
from Generals.ReconGenerals import ReconOption
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram
from Conversions.castor_id_to_histogram import castor_id_to_block_base_histogram, get_block_base_histogram_index


def get_scale_by_total_events(target_cdf, gpet_cdf, scan_time, scatter_histogram, scanner_option: ScannerOption, tof_option: TOFOption):
    # rr, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, -2], target_cdf[:, -1]
    num_of_randoms = get_randoms_histogram(target_cdf=target_cdf, scan_time=scan_time, scanner_option=scanner_option, tof_option=tof_option).sum()
    ratio = 1 / gpet_cdf.shape[0] * (target_cdf.shape[0] - num_of_randoms)
    return scatter_histogram * ratio


def get_scale_by_events_num_of_each_tof_bin(target_cdf, gpet_cdf, scan_time, tof_option: TOFOption, scatter_sinogram, scanner_option: ScannerOption):
    # rr, tof, tof_bin_index, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, 2], target_cdf[:, 3], target_cdf[:, -2], target_cdf[:, -1]

    target_events_per_tof_bin = np.bincount(target_cdf[:, 3].astype(int), minlength=tof_option.tof_bin_num)
    gpet_events_per_tof_bin = np.bincount(gpet_cdf[:, 3].astype(int), minlength=tof_option.tof_bin_num)

    num_of_randoms = get_randoms_sinogram(target_cdf, scan_time, scanner_option, None).sum()
    target_events_per_tof_bin = target_events_per_tof_bin.astype(np.float32) - (num_of_randoms / ((tof_option.tof_bin_num - 1) / 2))
    target_events_per_tof_bin[target_events_per_tof_bin < 0] = 0

    scale = 1 / gpet_events_per_tof_bin * target_events_per_tof_bin
    np.nan_to_num(scale, copy=False, nan=1, posinf=1, neginf=1)
    scatter_sinogram = scatter_sinogram * scale[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    return scatter_sinogram


def get_randoms_sinogram(target_cdf, scan_time, scanner_option, tof_option: TOFOption = None):
    # rr, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, -2], target_cdf[:, -1]
    michelogram = np.zeros([scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins], dtype=np.float32)
    # miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option, np.column_stack((target_cdf[:, -2], target_cdf[:, -1])), False)
    miche_index, miche_rm_option = crystal_base_id_to_block_base_sinogram(np.column_stack((target_cdf[:, -2], target_cdf[:, -1])))
    michelogram[tuple(miche_index.astype(int).T)] = target_cdf[:, 1][~miche_rm_option]*scan_time
    if tof_option is not None:
        michelogram = np.repeat(michelogram[np.newaxis, :, :, :, :], tof_option.tof_bin_num, axis=0) / ((tof_option.tof_bin_num - 1) / 2)
    return michelogram


def get_randoms_histogram(target_cdf, scan_time, scanner_option: ScannerOption, tof_option: TOFOption):
    # rr, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, -2], target_cdf[:, -1]
    unique_LOR_num = scanner_option.crystal_num ** 2
    histogram = np.zeros([int(unique_LOR_num)], dtype=np.float32)
    histogram_index = get_block_base_histogram_index(cdf_data=target_cdf, scanner_option=scanner_option, tof_option=tof_option)
    # randoms are not related to tof
    histogram[histogram_index[:, 0]] = target_cdf[:, 1] * scan_time
    histogram = np.repeat(histogram[:, np.newaxis], tof_option.tof_bin_num, axis=1) / tof_option.tof_bin_num
    return histogram


def get_sinogram_mask(mumap, tof_option, scanner_option, device_id):
    recon_option = ReconOption(
        img_dim=np.asarray(mumap.shape[0]),
        voxel_size=np.array([1, 1, 1]),
        output_dir="",
        ex_cdf_path="",
        tx_cdf_path="",
        bx_cdf_path="",
        num_of_subsets=1,
        num_of_iterations=1,
        psf_option=None,
        scanner_option=scanner_option,
        device_id=device_id
    )
    projector = Projector(recon_option, scanner_option, None, tof_option, device_id)
    i, j = np.meshgrid(np.arange(scanner_option.crystal_num, dtype=np.uint32), np.arange(scanner_option.crystal_num, dtype=np.uint32), indexing='ij')
    mask = i < j
    full_lor_index = np.column_stack((i[mask], j[mask]))
    ai = projector.projection_forward_lors(mumap, full_lor_index[:, 0], full_lor_index[:, 1], False)
    ai = np.exp(-ai)
    michelogram = np.zeros([scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option, full_lor_index, get_ssrb=False)
    np.add.at(michelogram, tuple(miche_index.astype(int).T), ai[~miche_rm_option])

    mask = michelogram > 0.9
    mask = np.repeat(mask[np.newaxis, :, :, :, :], tof_option.tof_bin_num, axis=0)
    return mask


def get_scale_by_sinogram_edge_fit(target_cdf, gpet_cdf, scan_time, tof_option: TOFOption, scatter_sinogram, scanner_option: ScannerOption, mumap, device_id):
    # rr, tof, tof_bin_index, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, 2], target_cdf[:, 3], target_cdf[:, -2], target_cdf[:, -1]

    miche_index, _ = crystal_base_id_to_block_base_sinogram(target_cdf[:, -2:])
    target_sinogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    index = tuple(np.column_stack((target_cdf[:, 3], miche_index)).astype(int).T)
    np.add.at(target_sinogram, index, 1)

    miche_index, _ = crystal_base_id_to_block_base_sinogram(gpet_cdf[:, -2:])
    gpet_sinogram = np.zeros([tof_option.tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    index = tuple(np.column_stack((gpet_cdf[:, 3], miche_index)).astype(int).T)
    np.add.at(gpet_sinogram, index, 1)

    randoms = get_randoms_sinogram(target_cdf=target_cdf, scan_time=scan_time, scanner_option=scanner_option, tof_option=tof_option)
    target_sinogram -= randoms
    target_sinogram[target_sinogram < 0] = 0

    # gaussian filter with maximum recovery
    # target_max = target_sinogram.max()
    # gpet_max = gpet_sinogram.max()
    # filter_sigma = 1
    # print(filter_sigma)
    # target_sinogram = scipy.ndimage.gaussian_filter(target_sinogram, sigma=filter_sigma, order=0, mode='reflect', truncate=1.0, axes=(-2, -1))
    # gpet_sinogram = scipy.ndimage.gaussian_filter(gpet_sinogram, sigma=filter_sigma, order=0, mode='reflect', truncate=1.0, axes=(-2, -1))
    # target_sinogram = target_sinogram / target_sinogram.max() * target_max
    # gpet_sinogram = gpet_sinogram / gpet_sinogram.max() * gpet_max

    miche_mask = get_sinogram_mask(mumap, tof_option, scanner_option, device_id)
    target_sinogram[miche_mask == 0] = np.nan
    gpet_sinogram[miche_mask == 0] = np.nan

    def linear_model(x, a, b):
        return a * x + b
    bounds = ([0, -np.inf], [np.inf, np.inf])  # a≥0，b可任意

    for i in range(tof_option.tof_bin_num):
        x = gpet_sinogram[i, :, :, :, :]
        y = target_sinogram[i, :, :, :, :]
        flag = (~np.isnan(x)) & (~np.isnan(y)) & (x >= 0) & (y >= 0)
        x = x[flag]
        y = y[flag]
        if x.shape[0] == 0:
            continue
        if y.shape[0] == 0:
            scatter_sinogram[i, :, :, :, :] = np.zeros_like(scatter_sinogram[i, :, :, :, :])
        else:
            coef, _ = scipy.optimize.curve_fit(linear_model, x, y, bounds=bounds)
            scatter_sinogram[i, :, :, :, :] = scatter_sinogram[i, :, :, :, :] * coef[0]  # + coef[1]

        # fit_y = x*coef[0] + coef[1]
        # plt.plot(x, fit_y)
        # plt.scatter(x, y, s=10, marker=".", alpha=0.3)
        # plt.show(block=True)
    return scatter_sinogram


def get_scale_by_histogram_bin_counts_sum_all_tof(target_cdf, gpet_cdf, scan_time, scatter_histogram, scanner_option: ScannerOption, tof_option: TOFOption):
    target_histogram = castor_id_to_block_base_histogram(cdf_data=target_cdf, scanner_option=scanner_option, tof_option=tof_option)
    gpet_histogram = castor_id_to_block_base_histogram(cdf_data=gpet_cdf, scanner_option=scanner_option, tof_option=tof_option)

    randoms = get_randoms_histogram(target_cdf=target_cdf, scan_time=scan_time, scanner_option=scanner_option, tof_option=tof_option)
    target_histogram -= randoms
    target_histogram[target_histogram < 0] = 0

    scale_factor = target_histogram.sum(axis=1) / gpet_histogram.sum(axis=1)
    scatter_histogram *= scale_factor[:, np.newaxis]

    np.nan_to_num(scatter_histogram, copy=False, posinf=0, neginf=0, nan=0)
    return scatter_histogram


