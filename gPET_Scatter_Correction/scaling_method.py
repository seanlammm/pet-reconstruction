import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from gPET_Scatter_Correction.others import tof_blurring, cdf_to_block_base_sinogram, crystal_base_id_to_block_base_sinogram


def get_cdf_info(cdf_path, wrr, wtof):
    op_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if wtof:
        # op_dtype[1:1] = [('tof_resolution', 'f4')]
        op_dtype[1:1] = [('tof', 'f4')]
    if wrr:
        op_dtype[1:1] = [('random_rate', 'f4')]
    cdf = np.fromfile(cdf_path, dtype=op_dtype)

    castor_id_1 = cdf["castor_id_1"]
    castor_id_2 = cdf["castor_id_2"]
    tof = cdf["tof"]

    swap_flag = tof > 0
    castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_1[swap_flag].copy(), castor_id_2[swap_flag].copy(), -tof[swap_flag].copy()

    if wrr & wtof:
        return np.column_stack((cdf['random_rate'], castor_id_1, castor_id_2, tof))
    elif wtof:
        return np.column_stack((castor_id_1, castor_id_2, tof))
    else:
        assert "Not finish yet. "


def get_scale_by_total_events(target_cdf_path, gpet_cdf_path, scan_time, scatter_sinogram, scanner_option):
    target_cdf = get_cdf_info(target_cdf_path, 0, 1)
    gpet_cdf = get_cdf_info(gpet_cdf_path, 0, 1)
    ratio = 1 / gpet_cdf.shape[0] * target_cdf.shape[0]
    scatter_sinogram = scatter_sinogram * ratio
    return scatter_sinogram


def get_scale_by_events_num_of_each_tof_bin(target_cdf_path, gpet_cdf_path, scan_time, tof_option, scatter_sinogram, scanner_option):
    '''
    scale by number of events in each tof bin
    :param correction target cdf path: [rr, castor_id1, castor_id2, tof]
    :param gpet true+scatter cdf path: [castor_id1, castor_id2, tof]
    :param scan_time:
    :return:
    '''

    target_cdf = get_cdf_info(target_cdf_path, 0, 1)
    gpet_cdf = get_cdf_info(gpet_cdf_path, 0, 1)

    target_tof_bin_index, _ = tof_option.get_tof_bin_index(target_cdf[:, -1])
    gpet_tof_bin_index, _ = tof_option.get_tof_bin_index(gpet_cdf[:, -1])

    target_events_per_tof_bin = np.bincount(target_tof_bin_index, minlength=tof_option.tof_bin_num)
    gpet_events_per_tof_bin = np.bincount(gpet_tof_bin_index, minlength=tof_option.tof_bin_num)

    rr_in_all = np.zeros(scanner_option.crystal_per_layer**2)
    if target_cdf.shape[1] == 4:
        rr_in_all[target_cdf[:, 1].astype(np.uint64) * scanner_option.crystal_per_layer + target_cdf[:, 2].astype(np.uint64)] = target_cdf[:, 0]
    num_of_random = np.sum(rr_in_all * scan_time)
    target_events_per_tof_bin = target_events_per_tof_bin.astype(float) - (num_of_random / tof_option.tof_bin_num)

    scale = 1 / gpet_events_per_tof_bin * target_events_per_tof_bin
    np.nan_to_num(scale, copy=False, nan=1, posinf=1, neginf=1)
    scatter_sinogram = scatter_sinogram * scale[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    return scatter_sinogram


def get_randoms_sinogram(target_cdf, tof_option, scan_time):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    rr, castor_id_1, castor_id_2 = target_cdf[:, 0], target_cdf[:, 1], target_cdf[:, 2]

    michelogram = np.zeros([scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    miche_index, miche_rm_option = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
    np.add.at(michelogram, tuple(miche_index.astype(int).T), rr[~miche_rm_option]*scan_time)

    michelogram = np.repeat(michelogram[np.newaxis, :, :, :, :], tof_option.tof_bin_num, axis=0) / ((tof_option.tof_bin_num - 1) / 2)
    return michelogram


def get_scale_by_sinogram_edge_fit(target_cdf_path, gpet_cdf_path, scan_time, tof_option, scatter_sinogram, scanner_option):
    '''
    scale by sinogram edge linear fit in each tof bin
    :param correction target cdf path: [rr, castor_id1, castor_id2, tof]
    :param gpet scatter cdf path: [castor_id1, castor_id2, tof]
    :param scan_time:
    :return:
    '''

    target_sinogram = cdf_to_block_base_sinogram(cdf_path=target_cdf_path, tof_option=tof_option, with_blur=0, wrr=0)
    gpet_sinogram = cdf_to_block_base_sinogram(cdf_path=gpet_cdf_path, tof_option=tof_option, with_blur=1, wrr=0)

    target_cdf = get_cdf_info(cdf_path=target_cdf_path, wrr=1, wtof=1)
    randoms = get_randoms_sinogram(target_cdf, tof_option, scan_time)
    target_sinogram -= randoms
    target_sinogram[target_sinogram < 0] = 0

    # gaussian filter with maximum recovery
    target_max = target_sinogram.max()
    gpet_max = gpet_sinogram.max()
    target_sinogram = scipy.ndimage.gaussian_filter(target_sinogram, sigma=2, order=0, mode='reflect', truncate=1.0, axes=(-2, -1))
    gpet_sinogram = scipy.ndimage.gaussian_filter(gpet_sinogram, sigma=2, order=0, mode='reflect', truncate=1.0, axes=(-2, -1))
    target_sinogram = target_sinogram / target_sinogram.max() * target_max
    gpet_sinogram = gpet_sinogram / gpet_sinogram.max() * gpet_max

    target_sinogram[:, :, :, :, 15:45] = np.nan
    gpet_sinogram[:, :, :, :, 10:45] = np.nan

    def linear_model(x, a, b):
        return a * x + b
    bounds = ([0, -np.inf], [np.inf, np.inf])  # a≥0，b可任意

    for i in range(tof_option.tof_bin_num):
        x = target_sinogram[i, :, :, :, :]
        y = gpet_sinogram[i, :, :, :, :]
        flag = (~np.isnan(x)) & (~np.isnan(y)) & (x > 0) & (y > 0)
        x = x[flag]
        y = y[flag]
        if (y.shape[0] == 0) | (y.shape[0] == 0):
            continue
        else:
            coef, _ = scipy.optimize.curve_fit(linear_model, x, y, bounds=bounds)
            scatter_sinogram[i, :, :, :, :] = scatter_sinogram[i, :, :, :, :] * coef[0]  # + coef[1]

        # fit_y = x*coef[0] + coef[1]
        # plt.plot(x, fit_y)
        # plt.scatter(x, y, s=10, marker=".", alpha=0.3)
        # plt.show(block=True)
    return scatter_sinogram
