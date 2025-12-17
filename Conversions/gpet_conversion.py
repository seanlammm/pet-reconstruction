import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram


def crystal_base_id_to_block_base_sinogram(castor_ids):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    axi_ids = castor_ids // 480 // 10
    cir_ids = castor_ids % 480 // 8
    mb_castor_ids = axi_ids * 60 + cir_ids

    miche_index, miche_rm_option, ssrb_index, rm_option = castor_id_to_sinogram(scanner_option, mb_castor_ids, get_ssrb=True)
    return miche_index, miche_rm_option, ssrb_index, rm_option


def tof_blurring(tof, tof_resolution):
    np.random.seed(42)
    sigma = tof_resolution / (2 * np.sqrt(2 * np.log(2)))  # ≈ fwhm / 2.3548
    gaussian_rands = np.random.normal(loc=0, scale=sigma, size=tof.shape[0])
    tof += gaussian_rands
    return tof


def cdf_to_block_base_sinogram(cdf_path: str, tof_bin_num, with_blur=0, w_tof=1, scanner_option=0):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")
    # scanner_option = ScannerOption("PET_11panel_LD")
    if w_tof:
        ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    else:
        ip_type = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    cdf = np.fromfile(cdf_path, dtype=ip_type)
    if w_tof:
        tof = cdf['tof']
    castor_id_1 = cdf['castor_id_1']
    castor_id_2 = cdf['castor_id_2']

    miche_index, miche_rm_option, _, _ = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
    # miche_index, miche_rm_option, _, _ = castor_id_to_sinogram(scanner_option, np.column_stack((castor_id_1, castor_id_2)), True)

    if w_tof:
        tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
        positive_tof_flag = tof > 0
        temp = castor_id_1[positive_tof_flag].copy()
        castor_id_1[positive_tof_flag] = castor_id_2[positive_tof_flag]
        castor_id_2[positive_tof_flag] = temp
        tof[positive_tof_flag] *= -1

        tof = tof[~miche_rm_option]
        if with_blur:
            tof = tof_blurring(tof, 300)

        tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
        tof_bin_index[tof_bin_index < 0] = 0
        tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1

        michelogram = np.zeros([tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
        np.add.at(michelogram, tuple(np.column_stack((tof_bin_index, miche_index)).astype(int).T), 1)
        return michelogram

    else:
        pass


def cdf_to_crystal_base_sinogram_to_block_base_sinogram(cdf_path, tof_bin_num, with_blur=0):
    _,  castor_ids_in_unique, crystal_base_michelogram_counts = cdf_to_crystal_base_sinogram(cdf_path, tof_bin_num, with_blur=with_blur, w_tof=1)

    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_Module_Base")

    miche_index, miche_rm_option, _, _ = crystal_base_id_to_block_base_sinogram(castor_ids_in_unique[:, 1:])
    castor_ids_in_unique = castor_ids_in_unique[~miche_rm_option, :]

    index = tuple(np.column_stack((castor_ids_in_unique[:, 0], miche_index)).astype(int).T)
    michelogram = np.zeros([tof_bin_num, scanner_option.rings, scanner_option.rings, scanner_option.views, scanner_option.bins])
    np.add.at(michelogram, index, crystal_base_michelogram_counts)
    return michelogram


def cdf_to_crystal_base_sinogram(cdf_path: str, tof_bin_num, with_blur=0, w_tof=1, scanner_option=0, num_of_chunk=30):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
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
        miche_index, miche_rm_option, _, _ = castor_id_to_sinogram(scanner_option, unique_ids[:, 1:], True)

        unique_ids = unique_ids[~miche_rm_option, :]
        counts = counts[~miche_rm_option]
        index = tuple(np.column_stack((unique_ids[:, 0], miche_index)).astype(int).T)
        np.add.at(michelogram, index, counts)

        # michelogram = scipy.ndimage.gaussian_filter(michelogram, sigma=3, order=0, mode='reflect', truncate=4.0, axes=(-2, -1))
        michelogram_counts = michelogram[index]

    return michelogram, unique_ids, michelogram_counts


def plot_compare():
    gate_miche, gate_ssrb = cdf_to_sinogram(cdf_path="/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf")
    gpet_miche, gpet_ssrb = cdf_to_sinogram(cdf_path="/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_scatter_wtof.cdf")

    ratio = 1 / 100968713 * 61015496
    gpet_miche = np.ceil(gpet_miche * ratio)
    gate_plot = gate_miche.sum(axis=(1, 2))
    gpet_plot = gpet_miche.sum(axis=(1, 2))

    print(1)
    plt.figure(figsize=(18, 6))
    plot_tof = 9
    plt.subplot(121)
    plt.title("GATE Scatter: %d" % gate_plot[plot_tof, :, :].sum())
    plt.imshow(gate_plot[plot_tof, :, :])
    plt.colorbar()
    plt.subplot(122)
    plt.title("gPET Scatter: %d" % gpet_plot[plot_tof, :, :].sum())
    plt.imshow(gpet_plot[plot_tof, :, :])
    plt.colorbar()
    plt.show(block=True)

def read_coins(coins_path, time_resolution=None, time_window=1000):
    coins_type = [('particle_id', 'i4'),
                 ('panel_id', 'i4'),
                 ('module_id', 'i4'),
                 ('crystal_id', 'i4'),
                 ('site_id', 'i4'),
                 ('event_id', 'i4'),
                 ('global_time', 'f8'),
                 ('deposited_energy', 'f4'),
                 ('local_pos_x', 'f4'),
                 ('local_pos_y', 'f4'),
                 ('local_pos_z', 'f4'),
                 ('scatter_flag', 'i8')]
    coins = np.fromfile(coins_path, dtype=coins_type)
    panel_id = coins["panel_id"].astype(np.double)
    module_id = coins["module_id"].astype(np.double)
    crystal_id = coins["crystal_id"].astype(np.double)
    scatter_flag = coins["scatter_flag"].astype(np.double)
    global_time = coins["global_time"].astype(np.double)
    global_time = (global_time * 1e6).astype(float)  # convert 1e-6s(us) to 1e-12s(ps)

    if time_resolution:  # in ps
        global_time = (global_time // time_resolution) * time_resolution

    castor_ids = np.zeros(panel_id.shape[0], dtype=np.uint32)
    cir_castor_id = panel_id * 6 * 8 + module_id % 6 * 8 + crystal_id % 8
    # with panel at up-side
    cir_castor_id[cir_castor_id > 0] = 480 - cir_castor_id[cir_castor_id > 0]  # inverse
    cir_castor_id += (480 - (6 * 8 * 9 + 1))  # move 0 from right to left
    cir_castor_id[cir_castor_id >= 480] -= 480  # make cir in [0, 480)
    cir_castor_id -= (6 * 8 * 5)  # make start panel at -y
    cir_castor_id[cir_castor_id < 0] += 480  # make cir positive
    # with panel at down-side
    axi_castor_id = module_id // 6 * 10 + crystal_id // 8
    castor_ids = axi_castor_id * 480 + cir_castor_id

    castor_ids = castor_ids.reshape([-1, 2])
    scatter_flag = scatter_flag.reshape([-1, 2])
    global_time = global_time.reshape([-1, 2])
    tof = - (global_time[:, 0] - global_time[:, 1]).astype(np.float32)

    out_of_tw = np.abs(tof) > time_window
    castor_ids = castor_ids[~out_of_tw]
    scatter_flag = scatter_flag[~out_of_tw]
    tof = tof[~out_of_tw]

    scatter_flag = scatter_flag.sum(axis=1) > 0
    return np.column_stack((castor_ids, tof, scatter_flag)).astype(int)


def get_tof_scatter_ratio(coins_dir, output_dir, output_fn):
    # tof_bin_num = 11
    # tof_interval = np.linspace(-1100, 1100, tof_bin_num + 1)
    tof_bin_num = 10
    tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
    prompts_in_tofbin = np.zeros([tof_bin_num*57600*57600], dtype=np.float32)  # [tof*57600*57600 + id1*57600 + id2]{(-inf, -750), [-750, -500), [-500, -250), [-250, inf]}
    scatters_in_tofbin = np.zeros([tof_bin_num*57600*57600], dtype=np.float32)  # make sure tof negative by switch id position

    for i in tqdm(range(9999)):
        coins_path = "%s/output_%d/coincidences.dat" % (coins_dir, i)
        if not os.path.exists(coins_path):
            print("output %d not exist" % i)
            continue
        coins_info = read_coins(coins_path)

        pos_tof_flag = coins_info[:, 2] > 0
        swap = coins_info[pos_tof_flag, 0]
        coins_info[pos_tof_flag, 0] = coins_info[pos_tof_flag, 1]
        coins_info[pos_tof_flag, 1] = swap
        coins_info[pos_tof_flag, 2] *= -1

        tof_bin_index = np.digitize(coins_info[:, 2], bins=tof_interval) - 1
        tof_bin_index[tof_bin_index < 0] = 0
        tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1
        coins_info[:, 2] = tof_bin_index

        coins_info = coins_info.astype(np.int64)
        scatters = coins_info[coins_info[:, 3] == 1, :][:, :3]
        np.add.at(prompts_in_tofbin, ((coins_info[:, 2] * 57600**2) + coins_info[:, 0]*57600 + coins_info[:, 1]), 1)
        np.add.at(scatters_in_tofbin, ((scatters[:, 2] * 57600**2) + scatters[:, 0]*57600 + scatters[:, 1]), 1)

    # scatters_ratio = np.zeros_like(prompts_in_tofbin, dtype=np.float32)
    # scatters_ratio[prompts_in_tofbin > 0] = scatters_in_tofbin[prompts_in_tofbin > 0] / prompts_in_tofbin[prompts_in_tofbin > 0]

    # output_dir = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/scatter_ratio_output/"
    # scatters_ratio.tofile("%s/scatter_ratio_20251027_human_460m_it2_tofbin_10_in_neg_1000_to_0.raw" % output_dir)
    scatters_in_tofbin.tofile("%s/%s" % (output_dir, output_fn))


def get_non_tof_scatter_ratio(coins_dir, output_dir, output_fn):
    prompts_in_tofbin = np.zeros([57600*57600], dtype=np.float32)  # [tof*57600*57600 + id1*57600 + id2]{(-inf, -750), [-750, -500), [-500, -250), [-250, inf]}
    scatters_in_tofbin = np.zeros([57600*57600], dtype=np.float32)  # make sure tof negative by switch id position

    for i in tqdm(range(9999)):
        coins_path = "%s/output_%d/coincidences.dat" % (coins_dir, i)
        if not os.path.exists(coins_path):
            continue
        coins_info = read_coins(coins_path)

        positive_tof_flag = tof > 0
        temp = castor_id_1[positive_tof_flag].copy()
        castor_id_1[positive_tof_flag] = castor_id_2[positive_tof_flag]
        castor_id_2[positive_tof_flag] = temp
        tof[positive_tof_flag] *= -1

        scatters = coins_info[coins_info[:, 3] == 1, :][:, :3]
        np.add.at(prompts_in_tofbin, (coins_info[:, 0]*57600 + coins_info[:, 1]), 1)
        np.add.at(scatters_in_tofbin, (scatters[:, 0]*57600 + scatters[:, 1]), 1)

    scatters_ratio = np.zeros_like(prompts_in_tofbin, dtype=np.float32)
    scatters_ratio[prompts_in_tofbin > 0] = scatters_in_tofbin[prompts_in_tofbin > 0] / prompts_in_tofbin[prompts_in_tofbin > 0]
    # output_dir = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/scatter_ratio_output/"
    # scatters_ratio.tofile("%s/scatter_ratio_20251027_human_460m_it2_tofbin_10_in_neg_1000_to_0.raw" % output_dir)
    scatters_ratio.tofile("%s/%s" % (output_dir, output_fn))


def get_scatter_counts(scatter_cdf_path, tof_bins):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")

    ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    input_cdf = np.fromfile(scatter_cdf_path, dtype=ip_type)
    castor_id_1 = input_cdf['castor_id_1']
    castor_id_2 = input_cdf['castor_id_2']
    tof = input_cdf['tof']

    swap_flag = tof > 0
    temp = castor_id_1[swap_flag]
    castor_id_1[swap_flag] = castor_id_2[swap_flag]
    castor_id_2[swap_flag] = temp
    tof[swap_flag] *= -1

    tof_bin_num = tof_bins
    tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
    tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
    tof_bin_index[tof_bin_index < 0] = 0
    tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1
    # if array in np.float32, it will lead to wrong add result
    scatter_counts = np.zeros([tof_bin_num, scanner_option.crystal_per_layer, scanner_option.crystal_per_layer])
    np.add.at(scatter_counts, tuple(np.column_stack((tof_bin_index, castor_id_1, castor_id_2)).astype(int).T), 1)

    return scatter_counts.astype(np.float32)


def add_scf_to_tof_cdf(input_cdf_path, scatter_cdf_path, acf_path, nf_path, scan_time, output_dir, output_filename, scaled_ratio):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    # scanner_option = ScannerOption("PET_11panel_Module_Base")

    # ip_type = [('time', 'i4'), ('rr', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('rr', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    cdf = np.fromfile(input_cdf_path, dtype=ip_type)
    time = cdf["time"]
    castor_id_1 = cdf["castor_id_1"]
    castor_id_2 = cdf["castor_id_2"]
    rr = np.zeros_like(time)  # cdf["rr"] #np.zeros_like(time)
    tof = cdf["tof"]
    del cdf

    swap_flag = tof > 0
    temp = castor_id_1[swap_flag]
    castor_id_1[swap_flag] = castor_id_2[swap_flag]
    castor_id_2[swap_flag] = temp
    tof[swap_flag] *= -1

    tof_bin_num = 10
    tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
    tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
    tof_bin_index[tof_bin_index < 0] = 0
    tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1

# # #############################################################
    # temp code
    gpet_scatter_miche = cdf_to_block_base_sinogram("/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_scatter_wtof.cdf", tof_bin_num, 1)
    gate_scatter_miche = cdf_to_block_base_sinogram("/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf", tof_bin_num, 0)

    gpet_scatter_miche *= scaled_ratio

    miche_index, miche_rm_option, _, _ = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))

    scf_index = tuple(np.column_stack((tof_bin_index, miche_index)).astype(int).T)
    gate_scf = gate_scatter_miche[scf_index]
    gpet_scf = gpet_scatter_miche[scf_index]

    gate_scf_sum_in_tofbin = np.zeros(10)
    gpet_scf_sum_in_tofbin = np.zeros(10)
    for i in range(10):
        gate_scf_sum_in_tofbin[i] = gate_scf[tof_bin_index==i].sum()
        gpet_scf_sum_in_tofbin[i] = gpet_scf[tof_bin_index==i].sum()

    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    gate_sum_in_tof_bin = gate_scatter_miche.sum(axis=(1, 2, 3, 4))
    gpet_sum_in_tof_bin = gpet_scatter_miche.sum(axis=(1, 2, 3, 4))
    unique, counts = np.unique(tof_bin_index, return_counts=True)

    gate_sum_in_tof_bin = np.round(gate_sum_in_tof_bin / gate_sum_in_tof_bin.sum() * 100, 1)
    gpet_sum_in_tof_bin = np.round(gpet_sum_in_tof_bin / gpet_sum_in_tof_bin.sum() * 100, 1)
    counts = np.round(counts / counts.sum() * 100, 1)

    data = pd.DataFrame({
        "Category": ["TOFBin %d" % i for i in range(10)] * 3,  # 分类标签
        "Group": ["gate_scatter"] * 10 + ["gpet_scatter"] * 10 + ["gate_target_events"] * 10,  # 分组标签（循环）
        "Percentage": list(gate_sum_in_tof_bin) + list(gpet_sum_in_tof_bin) + list(counts)  # 对应数值
    })
    data = pd.DataFrame({
        "Category": ["TOFBin %d" % i for i in range(10)] * 2,  # 分类标签
        "Group": ["gate_scatter"] * 10 + ["gpet_scatter"] * 10,  # 分组标签（循环）
        "Percentage": list((gate_scf_sum_in_tofbin/6400).astype(int)) + list((gpet_scf_sum_in_tofbin/6400).astype(int))  # 对应数值
    })

    plt.figure(figsize=(18, 8))

    # 核心：hue 参数指定分组列
    ax = sns.barplot(
        x="Category",  # X轴：分类
        y="Percentage",  # Y轴：数值
        hue="Group",  # 分组依据（实现并列）
        data=data,
        width=0.9,  # 每组整体宽度（0~1，避免分类重叠）
        edgecolor="black"  # 条形边框（增强可读性）
    )

    # 添加数值标签（关键：提升图表可读性）
    for container in ax.containers:
        ax.bar_label(container, fontsize=10, padding=5, fmt='%d')  # padding：标签与条形顶部距离

    # 自动调整布局，避免标签截断
    plt.tight_layout()
    plt.show(block=True)

    crystal_base_scatter_sinogram, _, _ = cdf_to_crystal_base_sinogram(scatter_cdf_path, tof_bin_num, 1)
    block_base_scatter_sinogram = cdf_to_crystal_base_sinogram_to_block_base_sinogram(scatter_cdf_path, tof_bin_num, with_blur=0)

    block_base_sino_index, block_base_rm_option, _, _ = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
    crystal_base_sino_index, crystal_base_rm_option, _, _ = castor_id_to_sinogram(scanner_option, np.column_stack((castor_id_1, castor_id_2)), True)

    block_base_tof = tof_bin_index[~block_base_rm_option]
    crystal_base_tof = tof_bin_index[~crystal_base_rm_option]

    block_base_index = tuple(np.column_stack((block_base_tof, block_base_sino_index)).astype(int).T)
    crystal_base_index = tuple(np.column_stack((crystal_base_tof, crystal_base_sino_index)).astype(int).T)
    block_base_scf = block_base_scatter_sinogram[block_base_index]
    crystal_base_scf = crystal_base_scatter_sinogram[crystal_base_index]

    print(crystal_base_scatter_sinogram.sum())
    print(block_base_scatter_sinogram.sum())

    print(block_base_scf.sum())
    print(crystal_base_scf.sum())

#     #############################################################

    scatter_sinogram = cdf_to_sinogram(scatter_cdf_path, tof_bin_num, 0)
    sino_index, rm_option, _, _ = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
    # sino_index, rm_option, _, _ = castor_id_to_sinogram(scanner_option, np.column_stack((castor_id_1, castor_id_2)), True)
    # scaled_ratio = 1 / sum(simulation true+scatter) * sum(acquisition true+scatter)
    scatter_counts = scatter_sinogram * scaled_ratio / 80  # rescale to crystal level
    print(scatter_counts.sum())
    # print(scatter_counts.sum(axis=(1, 2)).astype(int))

    tof_bin_index = tof_bin_index[~rm_option]
    castor_id_1 = castor_id_1[~rm_option]
    castor_id_2 = castor_id_2[~rm_option]
    tof = tof[~rm_option]
    rr = rr[~rm_option]
    time = time[~rm_option]

    index = tuple(np.column_stack((tof_bin_index, sino_index)).astype(int).T)
    scf = scatter_counts[index]
    print(scf.sum())
    light_speed = 3e8
    tof_bin_width = (1000 / tof_bin_num) * 1e-12 * light_speed * 1000  # tof_bin_interval * (ps -- 1e-12 --> s) * light_speed * (m -- 1e3 --> mm)
    scf /= (tof_bin_width * scan_time)

    ac_factor = np.fromfile(acf_path, dtype=np.float32)
    norm_factor = np.ones_like(ac_factor)  # np.ones_like(ac_factor)  #np.fromfile(nf_path, dtype=np.float32)
    index = castor_id_1 * 57600 + castor_id_2
    acf = ac_factor[index]
    norm = norm_factor[index]
    del ac_factor, norm_factor

    num_of_counts = castor_id_1.shape[0]
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = time
    structured_array['castor_id_1'] = castor_id_1
    structured_array['castor_id_2'] = castor_id_2
    structured_array['rr'] = rr
    structured_array['scf'] = scf
    structured_array['acf'] = acf
    structured_array['norm'] = norm
    structured_array['tof'] = tof

    cdf_file = os.path.join(output_dir, (output_filename + ".cdf"))
    cdh_file = os.path.join(output_dir, (output_filename + ".cdh"))

    buffer = structured_array.tobytes()
    with open(cdf_file, 'ab') as cdf:
        cdf.write(buffer)

    with open(cdh_file, 'w') as file:
        print(f"Data filename: {output_filename}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): {scan_time}", file=file)  # Remember to change the time
        print(f"Scanner name: {scanner_option.scanner}", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"Random correction flag: {1}", file=file)
        print(f"Scatter correction flag: {1}", file=file)
        print(f"Attenuation correction flag: {1}", file=file)

        print(f"Normalization correction flag: {1}", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {2000}", file=file)

        print(f"\n", file=file)


def add_nontofscf_to_tof_cdf(input_cdf_path, sr_path, acf_path, nf_path, scan_time, output_dir, output_filename):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")

    # ip_type = [('time', 'i4'), ('rr', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('rr', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    cdf = np.fromfile(input_cdf_path, dtype=ip_type)
    time = cdf["time"]
    castor_id_1 = cdf["castor_id_1"]
    castor_id_2 = cdf["castor_id_2"]
    rr = np.zeros_like(time)# cdf["rr"] #np.zeros_like(time)
    tof = cdf["tof"]
    del cdf

    ac_factor = np.fromfile(acf_path, dtype=np.float32)
    norm_factor = np.ones_like(ac_factor)  #np.fromfile(nf_path, dtype=np.float32)
    index = castor_id_1 * 57600 + castor_id_2
    acf = ac_factor[index]
    norm = norm_factor[index]
    del ac_factor, norm_factor

    # make tof negative
    tof_bin_num = 10
    scatter_ratio = np.fromfile(sr_path, dtype=np.float32)
    # [tof*57600*57600 + id1*57600 + id2] {(-inf, -750), [-750, -500), [-500, -250), [-250, inf]}
    # count events on each lor, then multiply with scatter_ratio to get scatter counts
    scatter_counts = np.zeros([57600 * 57600], dtype=np.float32)
    np.add.at(scatter_counts, (castor_id_1 * 57600 + castor_id_2), 1)
    scatter_counts *= scatter_ratio
    light_speed = 3e8
    tof_bin_width = (1000 / tof_bin_num) * 1e-12 * light_speed * 1000  # tof_bin_interval * (ps -- 1e-12 --> s) * light_speed * (m -- 1e3 --> mm)
    scatter_counts /= (tof_bin_width * scan_time)

    index = castor_id_1.astype(np.uint64) * 57600 + castor_id_2.astype(np.uint64)
    scf = scatter_counts[index]

    num_of_counts = castor_id_1.shape[0]
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = time
    structured_array['castor_id_1'] = castor_id_1
    structured_array['castor_id_2'] = castor_id_2
    structured_array['rr'] = rr
    structured_array['scf'] = scf
    structured_array['acf'] = acf
    structured_array['norm'] = norm
    structured_array['tof'] = tof

    cdf_file = os.path.join(output_dir, (output_filename + ".cdf"))
    cdh_file = os.path.join(output_dir, (output_filename + ".cdh"))

    buffer = structured_array.tobytes()
    with open(cdf_file, 'ab') as cdf:
        cdf.write(buffer)

    with open(cdh_file, 'w') as file:
        print(f"Data filename: {output_filename}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): {scan_time}", file=file)  # Remember to change the time
        print(f"Scanner name: {scanner_option.scanner}", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"Random correction flag: {1}", file=file)
        print(f"Scatter correction flag: {1}", file=file)
        print(f"Attenuation correction flag: {1}", file=file)
        print(f"Normalization correction flag: {1}", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {2000}", file=file)

        print(f"\n", file=file)


def estimated_scatter_only_cdf(input_cdf_path, sr_path, scan_time, output_dir, output_filename):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")

    ip_type = [('time', 'i4'), ('rr', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    cdf = np.fromfile(input_cdf_path, dtype=ip_type)
    scatter_ratio = np.fromfile(sr_path, dtype=np.float32)
    tof_bin_num = 4
    tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)

    castor_id_1 = cdf["castor_id_1"]
    castor_id_2 = cdf["castor_id_2"]
    rr = cdf["rr"]
    tof = cdf["tof"]
    del cdf

    # make tof negative
    positive_tof_flag = tof > 0
    temp = castor_id_1[positive_tof_flag].copy()
    castor_id_1[positive_tof_flag] = castor_id_2[positive_tof_flag]
    castor_id_2[positive_tof_flag] = temp
    tof[positive_tof_flag] *= -1

    tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
    tof_bin_index[tof_bin_index < 0] = 0
    tof_bin_index[tof_bin_index >= 4] = 3

    # [tof*57600*57600 + id1*57600 + id2] {(-inf, -750), [-750, -500), [-500, -250), [-250, inf]}
    # count events on each lor, then multiply with scatter_ratio to get scatter counts
    scatter_counts = np.zeros([4 * 57600 * 57600], dtype=np.float32)
    np.add.at(scatter_counts, ((tof_bin_index * 57600 ** 2) + castor_id_1 * 57600 + castor_id_2), 1)
    scatter_counts *= scatter_ratio

    output_tof = np.array([-875, -625, -375, -125])
    positive_index = np.where(scatter_counts > 0)[0]
    tof_bin_index = positive_index // (57600*57600)
    castor_id_1 = (positive_index - (tof_bin_index*57600*57600)) // 57600
    castor_id_2 = (positive_index - (tof_bin_index*57600*57600)) % 57600
    counts = np.ceil(scatter_counts[positive_index]).astype(int)

    castor_id_1 = np.repeat(castor_id_1, counts)
    castor_id_2 = np.repeat(castor_id_2, counts)
    tof = np.repeat(output_tof[tof_bin_index], counts)

    num_of_counts = counts.sum()
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = 0
    structured_array['castor_id_1'] = castor_id_1
    structured_array['castor_id_2'] = castor_id_2
    structured_array['tof'] = tof

    cdf_file = os.path.join(output_dir, (output_filename + ".cdf"))
    cdh_file = os.path.join(output_dir, (output_filename + ".cdh"))

    buffer = structured_array.tobytes()
    with open(cdf_file, 'ab') as cdf:
        cdf.write(buffer)

    with open(cdh_file, 'w') as file:
        print(f"Data filename: {output_filename}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): {scan_time}", file=file)  # Remember to change the time
        print(f"Scanner name: {scanner_option.scanner}", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"Random correction flag: {0}", file=file)
        print(f"Scatter correction flag: {0}", file=file)
        print(f"Attenuation correction flag: {0}", file=file)
        print(f"Normalization correction flag: {0}", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {2000}", file=file)

        print(f"\n", file=file)

def coins_to_cdf(file_path, coin_type, with_scatter=True):
    '''

    Args:
        file_path:
        output_path:
        coin_type: 0-crystal base, 1-module base
        with_scatter: output cdf with scatter or not

    Returns:
        coins information: [castor_id_1, castor_id_2, tof]

    '''
    coins_type = [('particle_id', 'i4'),
                  ('panel_id', 'i4'),
                  ('module_id', 'i4'),
                  ('crystal_id', 'i4'),
                  ('site_id', 'i4'),
                  ('event_id', 'i4'),
                  ('global_time', 'f8'),
                  ('deposited_energy', 'f4'),
                  ('local_pos_x', 'f4'),
                  ('local_pos_y', 'f4'),
                  ('local_pos_z', 'f4'),
                  ('scatter_flag', 'i8')]
    coins = np.fromfile(file_path, dtype=coins_type)
    panel_id = coins["panel_id"].astype(np.double)
    module_id = coins["module_id"].astype(np.double)
    crystal_id = coins["crystal_id"].astype(np.double)
    scatter_flag = coins["scatter_flag"].astype(np.double)
    global_time = coins["global_time"].astype(np.double)
    global_time = (global_time * 1e6).astype(float)  # convert 1e-6s(us) to 1e-12s(ps)

    castor_ids = np.zeros(panel_id.shape[0], dtype=np.uint32)
    if coin_type == 0:
        cir_castor_id = panel_id * 6 * 8 + module_id % 6 * 8 + crystal_id % 8
        # with panel at up-side
        cir_castor_id[cir_castor_id > 0] = 480 - cir_castor_id[cir_castor_id > 0]  # inverse
        cir_castor_id += (480 - (6*8*9+1))  # move 0 from right to left
        cir_castor_id[cir_castor_id >= 480] -= 480  # make cir in [0, 480)
        cir_castor_id -= (6*8*5)  # make start panel at -y
        cir_castor_id[cir_castor_id < 0] += 480  # make cir positive

        # with panel at down-side
        axi_castor_id = module_id // 6 * 10 + crystal_id // 8
        castor_ids = axi_castor_id * 480 + cir_castor_id
    elif coin_type == 1:
        cir_castor_id = panel_id * 6 + module_id % 6
        cir_castor_id[cir_castor_id > 0] = 60 - cir_castor_id[cir_castor_id > 0]
        axi_castor_id = module_id // 6
        castor_ids = axi_castor_id * 60 + cir_castor_id

    castor_ids = castor_ids.reshape([-1, 2])
    scatter_flag = scatter_flag.reshape([-1, 2])
    global_time = global_time.reshape([-1, 2])
    tof = - (global_time[:, 0] - global_time[:, 1]).astype(np.float32)

    if not with_scatter:
        scatter_coins_flag = scatter_flag.sum(axis=1) > 0
        castor_ids = castor_ids[~scatter_coins_flag]
        tof = tof[~scatter_coins_flag]

    return np.column_stack((castor_ids, tof))


def get_cdf(ip_path, op_path, op_file_name, w_scatter, w_tof, time_window):
    num_of_counts = 0
    if w_tof:
        cdf_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    else:
        cdf_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if not os.path.exists(op_path):
        os.makedirs(op_path)

    # output_path = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf/"
    # file_name = "simulation_cylinder_wscatter"
    output_cdf_path = op_path + op_file_name + ".cdf"
    output_cdh_path = op_path + op_file_name + ".cdh"

    tof_range = np.zeros(2)
    for i in tqdm(range(99999)):
        # file_path = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/output/output_%d/coincidences.dat" % i
        file_path = ip_path + "/output_%d/coincidences.dat" % i
        if not os.path.exists(file_path):
            continue
        coins_info = read_coins(file_path, time_resolution=None, time_window=time_window)
        if not w_scatter:
            coins_info = coins_info[coins_info[:, 3] == 0, :]

        structured_array = np.empty(coins_info.shape[0], dtype=cdf_dtype)
        structured_array['time'] = 0
        structured_array['castor_id_1'] = coins_info[:, 0]
        structured_array['castor_id_2'] = coins_info[:, 1]
        if w_tof:
            structured_array['tof'] = coins_info[:, 2]
            tof_range[0] = coins_info[:, 2].min() if coins_info[:, 2].min() < tof_range[0] else tof_range[0]
            tof_range[1] = coins_info[:, 2].max() if coins_info[:, 2].max() > tof_range[1] else tof_range[1]
        buffer = structured_array.tobytes()
        with open(output_cdf_path, 'ab') as cdf:
            cdf.write(buffer)
        num_of_counts += coins_info.shape[0]

    with open(output_cdh_path, 'w') as file:
        print(f"Data filename: {op_file_name}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): 1", file=file)
        print(f"Scanner name: PET_11panel_LD", file=file)
        print(f"Calibration factor: 1", file=file)
        print(f"lsotope: unknown", file=file)
        if w_tof:
            print(f"TOF information flag: {1}", file=file)
            print(f"TOF resolution (ps): {300}", file=file)
            print(f"Per event TOF resolution flag: {0}", file=file)
            print(f"List TOF measurement range (ps): {tof_range[1] - tof_range[0]}", file=file)
        print(f"\n", file=file)


def get_scatter_cdf(ip_path, op_path, op_file_name, w_tof):
    num_of_counts = 0
    if w_tof:
        cdf_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    else:
        cdf_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    if not os.path.exists(op_path):
        os.makedirs(op_path)

    # output_path = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf/"
    # file_name = "simulation_cylinder_wscatter"
    output_cdf_path = op_path + op_file_name + ".cdf"
    output_cdh_path = op_path + op_file_name + ".cdh"

    tof_range = np.zeros(2)
    for i in tqdm(range(99999)):
        file_path = ip_path + "/output_%d/coincidences.dat" % i
        if not os.path.exists(file_path):
            continue
        coins_info = read_coins(file_path, time_resolution=None, time_window=1000)
        coins_info = coins_info[coins_info[:, 3] == 1, :]


        structured_array = np.empty(coins_info.shape[0], dtype=cdf_dtype)
        structured_array['time'] = 0
        structured_array['castor_id_1'] = coins_info[:, 0]
        structured_array['castor_id_2'] = coins_info[:, 1]
        if w_tof:
            structured_array['tof'] = coins_info[:, 2]
            tof_range[0] = coins_info[:, 2].min() if coins_info[:, 2].min() < tof_range[0] else tof_range[0]
            tof_range[1] = coins_info[:, 2].max() if coins_info[:, 2].max() > tof_range[1] else tof_range[1]
        buffer = structured_array.tobytes()
        with open(output_cdf_path, 'ab') as cdf:
            cdf.write(buffer)
        num_of_counts += coins_info.shape[0]

    with open(output_cdh_path, 'w') as file:
        print(f"Data filename: {op_file_name}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): 1", file=file)
        print(f"Scanner name: PET_11panel_LD", file=file)
        print(f"Calibration factor: 1", file=file)
        print(f"lsotope: unknown", file=file)
        if w_tof:
            print(f"TOF information flag: {1}", file=file)
            print(f"TOF resolution (ps): {300}", file=file)
            print(f"Per event TOF resolution flag: {0}", file=file)
            print(f"List TOF measurement range (ps): {tof_range[1] - tof_range[0]}", file=file)
        print(f"\n", file=file)



def modified_lm_cdf():
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")

    ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    # non tof sc_factor
    input_cdf = np.fromfile("/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf", dtype=ip_type)
    castor_id_1 = input_cdf['castor_id_1']
    castor_id_2 = input_cdf['castor_id_2']
    tof = input_cdf['tof']
    swap_flag = tof > 0
    temp = castor_id_1[swap_flag]
    castor_id_1[swap_flag] = castor_id_2[swap_flag]
    castor_id_2[swap_flag] = temp
    tof[swap_flag] *= -1

    tof_bin_num = 10
    tof_interval = np.linspace(-1000, 0, tof_bin_num + 1)
    tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
    tof_bin_index[tof_bin_index < 0] = 0
    tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1
    scatter_counts = np.zeros([tof_bin_num * scanner_option.crystal_per_layer**2], dtype=np.float32)
    np.add.at(scatter_counts, ((tof_bin_index * scanner_option.crystal_per_layer ** 2) + castor_id_1 * scanner_option.crystal_per_layer + castor_id_2), 1)

    light_speed = 3e8
    tof_bin_width = (1000 / tof_bin_num) * 1e-12 * light_speed * 1000  # tof_bin_interval(ps -- 1e-12 --> s) * light_speed * (m -- 1e3 --> mm)
    sc_factor = scatter_counts / (tof_bin_width * 120)

    input_cdf = np.fromfile("/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf", dtype=ip_type)
    time = input_cdf['time']
    castor_id_1 = input_cdf['castor_id_1']
    castor_id_2 = input_cdf['castor_id_2']
    tof = input_cdf['tof']
    #
    # input_cdf = np.fromfile("/share/home/lyj/files/gate_script/WBBrain_cylinder_72_radius_source/tof_data/root_output/scatters.cdf", dtype=ip_type)
    # time = np.concatenate((time, input_cdf['time']))
    # castor_id_1 = np.concatenate((castor_id_1, input_cdf['castor_id_1']))
    # castor_id_2 = np.concatenate((castor_id_2, input_cdf['castor_id_2']))
    # tof = np.concatenate((tof, input_cdf['tof']))

    swap_flag = tof > 0
    temp = castor_id_1[swap_flag]
    castor_id_1[swap_flag] = castor_id_2[swap_flag]
    castor_id_2[swap_flag] = temp
    tof[swap_flag] *= -1

    tof_bin_index = np.digitize(tof, bins=tof_interval) - 1
    tof_bin_index[tof_bin_index < 0] = 0
    tof_bin_index[tof_bin_index >= tof_bin_num] = tof_bin_num - 1
    scf = sc_factor[((tof_bin_index * scanner_option.crystal_per_layer ** 2) + castor_id_1 * scanner_option.crystal_per_layer + castor_id_2)]

    ac_factor = np.fromfile("/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw", dtype=np.float32)
    acf = ac_factor[castor_id_1 * scanner_option.crystal_per_layer + castor_id_2]

    num_of_counts = castor_id_1.shape[0]
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = time
    structured_array['castor_id_1'] = castor_id_1
    structured_array['castor_id_2'] = castor_id_2
    structured_array['acf'] = acf
    structured_array['norm'] = 1
    structured_array['scf'] = scf
    structured_array['tof'] = tof

    output_dir = "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/"
    output_filename = "trues_and_scatters_tof_wacf_wnorm_wgatescf"
    cdf_file = os.path.join(output_dir, (output_filename + ".cdf"))
    cdh_file = os.path.join(output_dir, (output_filename + ".cdh"))

    buffer = structured_array.tobytes()
    with open(cdf_file, 'ab') as cdf:
        cdf.write(buffer)

    with open(cdh_file, 'w') as file:
        print(f"Data filename: {output_filename}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): {120}", file=file)  # Remember to change the time
        print(f"Scanner name: {scanner_option.scanner}", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"Random correction flag: {0}", file=file)
        print(f"Scatter correction flag: {1}", file=file)
        print(f"Attenuation correction flag: {1}", file=file)
        print(f"Normalization correction flag: {1}", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {2000}", file=file)

        print(f"\n", file=file)


if __name__ == "__main__":
    crystal_base_scatter_sinogram, _, _ = cdf_to_crystal_base_sinogram(
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/simulation_cylinder_scatter_wtof.cdf",
        10, 1, num_of_chunk=300)
    np.save(
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/scatters_wblur_tof_in_crystals_base",
        crystal_base_scatter_sinogram)

    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/sim_30b_events/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/",
    #     "simulation_cylinder_wscatter_wtof",
    #     True,
    #     True,
    #     1000
    # )
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/sim_30b_events/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/",
    #     "simulation_cylinder_woscatter_wtof",
    #     False,
    #     True,
    #     1000
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/sim_30b_events/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/",
    #     "simulation_cylinder_scatter_wotof",
    #     False
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/sim_30b_events/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/",
    #     "simulation_cylinder_scatter_wtof",
    #     True
    # )
    # plot_compare()
    # get_tof_scatter_ratio(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/output_100m/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/scatter_ratio_output/",
    #     "scatter_counts_simulation_cylinder_wtof_22_bins_neg_1200_to_0.raw"
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_30b_events/simulation_cylinder_scatter_wtof.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgpettofscf_1000ps_10bins_block_base_miche",
    #     1 / 100968713 * 61015496
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgatetofscf_1000ps_10bins_block_base_miche",
    #     1
    # )
    # modified_lm_cdf()

    # #################################################
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/output_fix_600m/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/",
    #     "20251027_human_wscatter_wtof",
    #     True,
    #     True
    # )
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/output_fix_600m/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/",
    #     "20251027_human_woscatter_wtof",
    #     False,
    #     True
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/output_fix_600m/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/",
    #     "251027_human_scatter_wtof",
    #     True
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/11panel_recon/20251027_human/20251027_t4_human_wrr_wtof_wengwin_410_610_tw1000ps.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/251027_human_scatter_wtof.cdf",
    #     "/share/home/lyj/files/11panel_recon/20251027_human/ac_factor_20251027_human.raw",
    #     "/share/home/lyj/files/11panel_recon/20251028_cylinder_norm/20251028_norm_factor.raw",
    #     1500,
    #     "/share/home/lyj/files/11panel_recon/20251027_human/",
    #     "20251027_t4_human_wrr_wtof_wacf_wnorm_wgpettofscf_1000ps_10bins_block_base_miche",
    #     1 / 121483712 * 467513723
    # )