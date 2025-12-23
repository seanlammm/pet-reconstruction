import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys
import scipy
sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction")
from Generals.ScannerGenerals import ScannerOption
from Generals.TOFGenerals import TOFOption
from Conversions.castor_id_to_sinogram import castor_id_to_sinogram
from gPET_Scatter_Correction.others import crystal_base_id_to_block_base_sinogram, tof_blurring, cdf_to_block_base_sinogram, cdf_to_crystal_base_sinogram_to_block_base_sinogram, cdf_to_crystal_base_sinogram
from gPET_Scatter_Correction.scaling_method import get_scale_by_sinogram_edge_fit, get_scale_by_total_events, get_scale_by_events_num_of_each_tof_bin


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
    os.chdir(r"/")
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


def add_scf_to_tof_cdf(input_cdf_path, gpet_prompt_cdf_path, gpet_scatter_cdf_path, acf_path, nf_path, scan_time, output_dir, output_filename):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    tof_option = TOFOption(tof_resolution=300, tof_bin_num=21, tof_range_in_ps=2000)

    # scanner_option = ScannerOption("PET_11panel_Module_Base")

    ip_type = [('time', 'i4'), ('rr', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    # ip_type = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('rr', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    cdf = np.fromfile(input_cdf_path, dtype=ip_type)
    time = cdf["time"]
    castor_id_1 = cdf["castor_id_1"]
    castor_id_2 = cdf["castor_id_2"]
    rr = cdf["rr"]  # cdf["rr"] #np.zeros_like(time)
    tof = cdf["tof"]
    del cdf

    swap_flag = tof > 0
    castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_1[swap_flag].copy(), castor_id_2[swap_flag].copy(), -tof[swap_flag].copy()

    tof_bin_index, _ = tof_option.get_tof_bin_index(tof)
    scatter_sinogram = cdf_to_block_base_sinogram(cdf_path=gpet_scatter_cdf_path, tof_option=tof_option, with_blur=1)
    sino_index, rm_option = crystal_base_id_to_block_base_sinogram(np.column_stack((castor_id_1, castor_id_2)))
    # scatter_counts = get_scale_by_total_events(input_cdf_path, gpet_prompt_cdf_path, scan_time, scatter_sinogram, scanner_option) / 6400
    # scatter_counts = get_scale_by_events_num_of_each_tof_bin(input_cdf_path, gpet_prompt_cdf_path, scan_time, tof_option, scatter_sinogram, scanner_option) / 6400
    scatter_counts = get_scale_by_sinogram_edge_fit(input_cdf_path, gpet_prompt_cdf_path, scan_time, tof_option, scatter_sinogram, scanner_option) / 6400
    print(scatter_counts.sum())

    tof_bin_index = tof_bin_index[~rm_option]
    castor_id_1 = castor_id_1[~rm_option]
    castor_id_2 = castor_id_2[~rm_option]
    tof = tof[~rm_option]
    rr = rr[~rm_option]
    time = time[~rm_option]

    index = tuple(np.column_stack((tof_bin_index, sino_index)).astype(int).T)
    scf = scatter_counts[index]
    print("gPET Scatter Sum: %f. " % scf.sum())
    scf /= (tof_option.tof_bin_width * scan_time)

    ac_factor = np.fromfile(acf_path, dtype=np.float32)
    norm_factor = np.fromfile(nf_path, dtype=np.float32)  # np.ones_like(ac_factor)  #np.fromfile(nf_path, dtype=np.float32)
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


if __name__ == "__main__":

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
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_wscatter_wtof.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_scatter_wtof.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgpettofscf_1000ps_tof_bin_in_mm_scale_by_sinogram_edge_wgfs3t1"
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgatetofscf_1000ps_tof_bin_in_mm_scale_by_sinogram_edge_wgfs1t1",
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
    add_scf_to_tof_cdf(
        "/share/home/lyj/files/11panel_recon/20251027_human/20251027_t4_human_wrr_wtof_wengwin_410_610_tw1000ps.cdf",
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/20251027_human_wscatter_wtof.cdf",
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_fix_600m/251027_human_scatter_wtof.cdf",
        "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
        "/share/home/lyj/files/11panel_recon/20251028_cylinder_norm/20251028_norm_factor.raw",
        1500,
        "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
        "trues_scatters_wtof_wacf_wnorm_wgpettofscf_1000ps_tof_bin_in_mm_scale_by_sinogram_edge_wgfs2t1",
    )
