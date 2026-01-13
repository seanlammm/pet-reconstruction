import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Generals.TOFGenerals import TOFOption
from gPET_Scatter_Correction.sinogram.others import crystal_base_id_to_block_base_sinogram
from Generals.Projector import Projector
from Generals.ReconGenerals import ReconOption
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram
from gPET_Scatter_Correction.sinogram.others import castor_id_to_block_base_sinogram


def get_scale_by_total_events(target_cdf, gpet_cdf, scan_time, scatter_sinogram, scanner_option: ScannerOption):
    # rr, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, -2], target_cdf[:, -1]
    num_of_randoms = get_randoms_sinogram(target_cdf, scan_time, scanner_option, None).sum()
    ratio = 1 / gpet_cdf.shape[0] * (target_cdf.shape[0] - num_of_randoms)
    return scatter_sinogram * ratio


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

    castor_ids = target_cdf[:, -2:]
    axi_ids = castor_ids // 480 // 10
    cir_ids = castor_ids % 480 // 8
    mb_castor_ids = axi_ids * 60 + cir_ids
    flag = mb_castor_ids[:, 0] > mb_castor_ids[:, 1]
    mb_castor_ids[flag, 0], mb_castor_ids[flag, 1] = mb_castor_ids[flag, 1].copy(), mb_castor_ids[flag, 0].copy()

    miche_index, miche_rm_option = castor_id_to_sinogram(scanner_option=scanner_option, castor_ids=mb_castor_ids, get_ssrb=False)
    michelogram[tuple(miche_index.astype(int).T)] = target_cdf[:, 1][~miche_rm_option]*scan_time
    if tof_option is not None:
        michelogram = np.repeat(michelogram[np.newaxis, :, :, :, :], tof_option.tof_bin_num, axis=0) / tof_option.tof_bin_num
    return michelogram


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


def get_scale_by_sinogram_edge_fit(target_cdf, target_sino_index, gpet_sino_index, scan_time, tof_option: TOFOption, scatter_sinogram, scanner_option: ScannerOption, mumap, device_id):
    # rr, tof, tof_bin_index, castor_id_1, castor_id_2 = target_cdf[:, 1], target_cdf[:, 2], target_cdf[:, 3], target_cdf[:, -2], target_cdf[:, -1]
    target_sinogram = sinogram_index_to_block_base_sinogram(sinogram_index=target_sino_index, scanner_option=scanner_option, tof_option=tof_option)
    gpet_sinogram = sinogram_index_to_block_base_sinogram(sinogram_index=gpet_sino_index, scanner_option=scanner_option, tof_option=tof_option)

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


def get_scale_by_sinogram_bin_counts_sum_all_tof(target_cdf, gpet_prompt_cdf, scan_time, scatter_sinogram, scanner_option: ScannerOption, tof_option: TOFOption):
    target_sinogram = castor_id_to_block_base_sinogram(cdf_data=target_cdf, scanner_option=scanner_option, tof_option=tof_option)
    gpet_sinogram = castor_id_to_block_base_sinogram(cdf_data=gpet_prompt_cdf, scanner_option=scanner_option, tof_option=tof_option)

    target_sinogram = target_sinogram.sum(axis=0)
    gpet_sinogram = gpet_sinogram.sum(axis=0)

    randoms = get_randoms_sinogram(target_cdf=target_cdf, scan_time=scan_time, scanner_option=scanner_option)
    target_sinogram -= randoms
    target_sinogram[target_sinogram < 0] = 0
    scale_factor = target_sinogram.sum(axis=(2, 3)) / gpet_sinogram.sum(axis=(2, 3))

    sigma = 8 / (2 * np.sqrt(2 * np.log(2)))
    sum_before_filter = scatter_sinogram.sum(axis=(3, 4))
    scatter_sinogram = scipy.ndimage.gaussian_filter(scatter_sinogram, sigma=sigma, mode='reflect', axes=(-2, -1))
    sum_after_filter = scatter_sinogram.sum(axis=(3, 4))
    scatter_sinogram = scatter_sinogram / sum_after_filter[:, :, :, np.newaxis, np.newaxis] * sum_before_filter[:, :, :, np.newaxis, np.newaxis]
    np.nan_to_num(scatter_sinogram, copy=False, posinf=0, neginf=0, nan=0)

    scatter_sinogram *= scale_factor[np.newaxis, :, :, np.newaxis, np.newaxis]
    return scatter_sinogram


