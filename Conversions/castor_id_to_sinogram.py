import numpy as np
from tqdm import tqdm

from Generals.ScannerGenerals import ScannerOption


def castor_id_to_sinogram(scanner_option: ScannerOption, castor_ids, get_ssrb=False):
    tang_bins = scanner_option.crystal_per_ring - scanner_option.submodules_xy * scanner_option.crystals_xy

    ring1, ring2 = castor_ids[:, 0] // scanner_option.crystal_per_ring, castor_ids[:, 1] // scanner_option.crystal_per_ring
    crystal1, crystal2 = castor_ids[:, 0] % scanner_option.crystal_per_ring, castor_ids[:, 1] % scanner_option.crystal_per_ring

    phi = ((crystal1 + crystal2 + scanner_option.crystal_per_ring // 2) % scanner_option.crystal_per_ring) // 2

    delta = np.abs(crystal1 - crystal2)
    u = np.zeros_like(phi)
    option = ((crystal1 + crystal2) < (3 * scanner_option.crystal_per_ring // 2)) & ((crystal1 + crystal2) >= (scanner_option.crystal_per_ring // 2))
    u[option] = delta[option] - scanner_option.crystal_per_ring // 2 + tang_bins // 2
    u[~option] = -delta[~option] + scanner_option.crystal_per_ring // 2 + tang_bins // 2

    miche_rm_option = (u >= tang_bins) | (u < 0)
    u = np.delete(u, miche_rm_option)
    ring1 = np.delete(ring1, miche_rm_option)
    ring2 = np.delete(ring2, miche_rm_option)
    crystal1 = np.delete(crystal1, miche_rm_option)
    crystal2 = np.delete(crystal2, miche_rm_option)
    phi = np.delete(phi, miche_rm_option)

    zi = np.zeros_like(crystal1)
    option_1 = (u % 2 == 0)
    zi[option_1] = (scanner_option.crystal_per_ring // 2 - (crystal1[option_1] - crystal2[option_1]) - 1) // 2
    zi[~option_1] = (scanner_option.crystal_per_ring // 2 - (crystal1[~option_1] - crystal2[~option_1])) // 2

    option_2 = (zi >= scanner_option.crystal_per_ring // 4) & option_1
    option_3 = (zi <= -(scanner_option.crystal_per_ring // 4)) & option_1
    zi[option_2] = zi[option_2] - scanner_option.crystal_per_ring // 2 + 1
    zi[option_3] = zi[option_3] + scanner_option.crystal_per_ring // 2 - 1

    option_4 = (zi >= scanner_option.crystal_per_ring // 4) & ~option_1
    option_5 = (zi <= -(scanner_option.crystal_per_ring // 4)) & ~option_1
    zi[option_4] = zi[option_4] - scanner_option.crystal_per_ring // 2
    zi[option_5] = zi[option_5] + scanner_option.crystal_per_ring // 2

    c1 = crystal1 + zi
    c2 = crystal2 - zi

    option_1 = c1 >= scanner_option.crystal_per_ring
    c1[option_1] = c1[option_1] - scanner_option.crystal_per_ring
    option_2 = c1 < 0
    c1[option_2] = c1[option_2] + scanner_option.crystal_per_ring

    option_1 = c2 >= scanner_option.crystal_per_ring
    c2[option_1] = c2[option_1] - scanner_option.crystal_per_ring
    option_2 = c2 < 0
    c2[option_2] = c2[option_2] + scanner_option.crystal_per_ring

    flip_flag = c1 > c2
    swap = ring1[flip_flag].copy()
    ring1[flip_flag] = ring2[flip_flag].copy()
    ring2[flip_flag] = swap

    miche_index = np.column_stack((ring1, ring2, phi, u)).astype(np.uint32)

    if get_ssrb:
        ssrb_index, ssrb_rm_option = michelogram_to_ssrb(scanner_option, miche_index)
        rm_option = miche_rm_option.copy()
        rm_option[~miche_rm_option] = ssrb_rm_option
        return miche_index, miche_rm_option, ssrb_index, rm_option
    else:
        return miche_index, miche_rm_option


def michelogram_to_ssrb(scanner_option: ScannerOption, miche_index):
    ring1 = miche_index[:, 0].astype(int)
    ring2 = miche_index[:, 1].astype(int)
    ring_difference = ring1 - ring2
    max_ring_diff = scanner_option.crystals_z * scanner_option.submodules_z * scanner_option.rsector_z * scanner_option.modules_z - 1
    axial_rebinned = np.floor((((ring1 + 1 + ring2 + 1) / 2 - 0.5) / 0.5) - 1)

    rm_option = np.abs(ring_difference) > max_ring_diff
    ssrb_index = np.column_stack((
        axial_rebinned[~rm_option], miche_index[~rm_option, 2:],
    )).astype(np.uint32)
    return ssrb_index, rm_option


def assign_value(scanner_option, use_ssrb, sinogram_index, coin_counts):
    if use_ssrb:
        sinogram = np.zeros([2 * scanner_option.max_ring_diff - 1, scanner_option.views, scanner_option.bins])
    else:
        sinogram = np.zeros(shape=(scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins))

    unique, inverse, counts = np.unique(sinogram_index, axis=0, return_counts=True, return_inverse=True)
    sums = np.bincount(inverse, weights=coin_counts)
    sinogram[tuple(unique.T)] += sums

    if np.sum(sinogram) != np.sum(coin_counts):
        raise Exception("Counts do not match. ")
    return sinogram


def get_sinogram(scanner_option: ScannerOption, coins=None, get_ssrb=False):
    michelogram = np.zeros(shape=(scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins))
    ssrb_sinogram = np.zeros([2 * scanner_option.max_ring_diff - 1, scanner_option.views, scanner_option.bins])
    if coins is not None:
        infos = castor_id_to_sinogram(scanner_option, coins[:, :2], get_ssrb)
        if get_ssrb:
            ssrb_sinogram += assign_value(scanner_option, True, infos[2], coins[~infos[3], 2])
        else:
            michelogram += assign_value(scanner_option, False, infos[0], coins[~infos[1], 2])

    return michelogram, ssrb_sinogram


def check_id_in_sinogram(scanner_option: ScannerOption, sinogram):
    lac_info = []
    get_ssrb = True if sinogram.ndim == 3 else False

    for i in tqdm(range(scanner_option.crystal_per_layer), desc="Checking the effective lac location..."):
        current_id_comb = np.column_stack((np.repeat(i, scanner_option.crystal_per_layer - i), np.arange(i, scanner_option.crystal_per_layer)))
        infos = castor_id_to_sinogram(scanner_option, current_id_comb, get_ssrb)
        current_sinogram_index, rm_index = infos[-2:]
        current_id_comb = np.delete(current_id_comb, rm_index, axis=0)
        current_sinogram_counts = sinogram[tuple(current_sinogram_index.T)]
        zero_index = current_sinogram_counts == 0
        if np.sum(zero_index) == current_id_comb.shape[0]:
            continue
        # id1-id2-counts-phi
        lac_info.append(np.column_stack((current_id_comb, current_sinogram_counts, current_sinogram_index[:, 2]))[~zero_index, :])

    lac_info = np.concatenate(lac_info, axis=0)
    return lac_info

