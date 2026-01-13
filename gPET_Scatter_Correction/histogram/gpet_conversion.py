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
from gPET_Scatter_Correction.histogram.others import crystal_base_id_to_block_base_sinogram, tof_blurring, cdf_to_block_base_sinogram, cdf_to_crystal_base_sinogram, get_cdf_info
from gPET_Scatter_Correction.histogram.scaling_method import get_scale_by_sinogram_edge_fit, get_scale_by_total_events, get_scale_by_events_num_of_each_tof_bin, get_scale_by_histogram_bin_counts_sum_all_tof
from Conversions.castor_id_to_histogram import castor_id_to_block_base_histogram, get_block_base_histogram_index, castor_id_to_crystal_base_histogram, get_crystal_base_histogram_index


def read_coins(coins_path, time_window=1000):
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
    tof = (global_time[:, 0] - global_time[:, 1]).astype(np.float32)

    castor_id_1, castor_id_2 = castor_ids[:, 0], castor_ids[:, 1]
    swap_flag = castor_id_1 > castor_id_2
    castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_2[swap_flag].copy(), castor_id_1[swap_flag].copy(), -tof[swap_flag].copy()
    tof = tof_blurring(tof=tof, tof_resolution=300)
    swap_flag = tof > 0
    castor_id_1[swap_flag], castor_id_2[swap_flag], tof[swap_flag] = castor_id_2[swap_flag].copy(), castor_id_1[swap_flag].copy(), -tof[swap_flag].copy()
    castor_ids = np.column_stack((castor_id_1, castor_id_2))

    out_of_tw = np.abs(tof) > time_window
    castor_ids = castor_ids[~out_of_tw]
    scatter_flag = scatter_flag[~out_of_tw]
    tof = tof[~out_of_tw]

    scatter_flag = scatter_flag.sum(axis=1) > 0
    return np.column_stack((castor_ids, tof, scatter_flag)).astype(int)


def add_scf_to_tof_cdf(input_cdf_path, gpet_prompt_cdf_path, gpet_scatter_cdf_path, acf_path, nf_path, scan_time, output_dir, output_filename, mumap):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    crystal_base_scanner_option = ScannerOption("PET_11panel_LD")
    module_base_scanner_option = ScannerOption("PET_11panel_Module_Base")
    tof_option = TOFOption(tof_resolution=300, tof_bin_num=21, tof_range_in_ps=2000)

    # [time, rr, tof, tof_bin_index, castor_id_1, castor_id_2]
    target_cdf = get_cdf_info(cdf_path=input_cdf_path, tof_option=tof_option, wrr=0, wtof=1)
    gpet_prompt_cdf = get_cdf_info(cdf_path=gpet_prompt_cdf_path, tof_option=tof_option, wrr=0, wtof=1)
    gpet_scatter_cdf = get_cdf_info(cdf_path=gpet_scatter_cdf_path, tof_option=tof_option, wrr=0, wtof=1)

    scatter_histogram = castor_id_to_block_base_histogram(cdf_data=gpet_scatter_cdf, scanner_option=module_base_scanner_option, tof_option=tof_option)
    # scatter_histogram = get_scale_by_total_events(target_cdf=target_cdf, gpet_cdf=gpet_prompt_cdf, scan_time=scan_time, scatter_histogram=scatter_histogram, scanner_option=module_base_scanner_option, tof_option=tof_option)
    # scatter_histogram = get_scale_by_histogram_bin_counts_sum_all_tof(target_cdf=target_cdf, gpet_cdf=gpet_prompt_cdf, scan_time=scan_time, scatter_histogram=scatter_histogram, scanner_option=module_base_scanner_option, tof_option=tof_option)
    scatter_histogram /= (6400)

    index = get_block_base_histogram_index(cdf_data=target_cdf, scanner_option=module_base_scanner_option, tof_option=tof_option)
    scf = scatter_histogram[tuple(index.T)]
    print("gPET Scatter Sum: %f. " % scf.sum())
    scf /= (tof_option.tof_bin_width * scan_time)

    ac_factor = np.fromfile(acf_path, dtype=np.float32)
    norm_factor = np.ones_like(ac_factor) if np.all(target_cdf[:, 1] == 0) else np.fromfile(nf_path, dtype=np.float32)
    index = (target_cdf[:, -2].astype(np.uint64) * crystal_base_scanner_option.crystal_num + target_cdf[:, -1].astype(np.uint64))
    acf = ac_factor[index]
    norm = norm_factor[index]
    del ac_factor, norm_factor

    num_of_counts = target_cdf.shape[0]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('rr', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = target_cdf[:, 0].astype(np.int32)
    structured_array['castor_id_1'] = target_cdf[:, -2].astype(np.int32)
    structured_array['castor_id_2'] = target_cdf[:, -1].astype(np.int32)
    structured_array['rr'] = target_cdf[:, 1].astype(np.float32)
    structured_array['scf'] = scf.astype(np.float32)
    structured_array['acf'] = acf.astype(np.float32)
    structured_array['norm'] = norm.astype(np.float32)
    structured_array['tof'] = target_cdf[:, 2].astype(np.float32)

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
        print(f"Scanner name: {crystal_base_scanner_option.scanner}", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"Random correction flag: {1}", file=file)
        print(f"Scatter correction flag: {1}", file=file)
        print(f"Attenuation correction flag: {1}", file=file)

        print(f"Normalization correction flag: {1}", file=file)
        print(f"TOF information flag: {1}", file=file)
        print(f"TOF resolution (ps): {300}", file=file)
        print(f"Per event TOF resolution flag: {0}", file=file)
        print(f"List TOF measurement range (ps): {2000}", file=file)
        print(f"List TOF quantization bin size (ps): {tof_option.tof_bin_width/tof_option.light_speed}", file=file)

        print(f"\n", file=file)


def add_scf_to_tof_cdf_crystal_base(input_cdf_path, gpet_prompt_cdf_path, gpet_scatter_cdf_path, acf_path, nf_path, scan_time, output_dir, output_filename, mumap):
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    crystal_base_scanner_option = ScannerOption("PET_11panel_LD")
    module_base_scanner_option = ScannerOption("PET_11panel_Module_Base")
    tof_option = TOFOption(tof_resolution=300, tof_bin_num=21, tof_range_in_ps=2000)

    # [time, rr, tof, tof_bin_index, castor_id_1, castor_id_2]
    target_cdf = get_cdf_info(cdf_path=input_cdf_path, tof_option=tof_option, wrr=0, wtof=1)
    gpet_prompt_cdf = get_cdf_info(cdf_path=gpet_prompt_cdf_path, tof_option=tof_option, wrr=0, wtof=1)
    gpet_scatter_cdf = get_cdf_info(cdf_path=gpet_scatter_cdf_path, tof_option=tof_option, wrr=0, wtof=1)

    # test_code
    block_base_scatter_histogram = castor_id_to_block_base_histogram(cdf_data=gpet_scatter_cdf, scanner_option=module_base_scanner_option, tof_option=tof_option)
    crystal_base_scatter_histogram = castor_id_to_crystal_base_histogram(cdf_data=gpet_scatter_cdf, scanner_option=crystal_base_scanner_option, tof_option=tof_option)

    index = get_crystal_base_histogram_index(cdf_data=target_cdf, scanner_option=crystal_base_scanner_option)
    scf_cb = crystal_base_scatter_histogram[tuple(index.T)]
    index = get_block_base_histogram_index(cdf_data=target_cdf, scanner_option=module_base_scanner_option)
    scf_bb = block_base_scatter_histogram[tuple(index.T)]/6400

    print("test")
    ####################################################
    scatter_histogram = castor_id_to_crystal_base_histogram(cdf_data=gpet_scatter_cdf, scanner_option=crystal_base_scanner_option, tof_option=tof_option)
    # scatter_histogram = get_scale_by_total_events(target_cdf=target_cdf, gpet_cdf=gpet_prompt_cdf, scan_time=scan_time, scatter_histogram=scatter_histogram, scanner_option=module_base_scanner_option, tof_option=tof_option)
    # scatter_counts = get_scale_by_sinogram_bin_counts_sum_all_tof(target_cdf=target_cdf, gpet_cdf=gpet_prompt_cdf, scan_time=scan_time, scatter_sinogram=scatter_sinogram, scanner_option=module_base_scanner_option)

    index = get_crystal_base_histogram_index(cdf_data=target_cdf, scanner_option=crystal_base_scanner_option)
    scf = scatter_histogram[tuple(index.T)]
    print("Scatter Sum: %f. " % scf.sum())
    scf /= (tof_option.tof_bin_width * scan_time)

    ac_factor = np.fromfile(acf_path, dtype=np.float32)
    norm_factor = np.ones_like(ac_factor) if np.all(target_cdf[:, 1] == 0) else np.fromfile(nf_path, dtype=np.float32)
    index = (target_cdf[:, -2].astype(np.uint64) * crystal_base_scanner_option.crystal_num + target_cdf[:, -1].astype(np.uint64))
    acf = ac_factor[index]
    norm = norm_factor[index]
    del ac_factor, norm_factor

    num_of_counts = target_cdf.shape[0]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('scf', 'f4'), ('rr', 'f4'), ('norm', 'f4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    structured_array = np.empty(num_of_counts, dtype=op_type)
    structured_array['time'] = target_cdf[:, 0].astype(np.int32)
    structured_array['castor_id_1'] = target_cdf[:, -2].astype(np.int32)
    structured_array['castor_id_2'] = target_cdf[:, -1].astype(np.int32)
    structured_array['rr'] = target_cdf[:, 1].astype(np.float32)
    structured_array['scf'] = scf.astype(np.float32)
    structured_array['acf'] = acf.astype(np.float32)
    structured_array['norm'] = norm.astype(np.float32)
    structured_array['tof'] = target_cdf[:, 2].astype(np.float32)

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
        print(f"Scanner name: {crystal_base_scanner_option.scanner}", file=file)
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

    output_cdf_path = op_path + op_file_name + ".cdf"
    output_cdh_path = op_path + op_file_name + ".cdh"

    tof_range = np.zeros(2)
    for i in tqdm(range(99999)):
        file_path = ip_path + "/output_%d/coincidences.dat" % i
        if not os.path.exists(file_path):
            continue
        coins_info = read_coins(file_path, time_window=time_window)
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
        coins_info = read_coins(file_path, time_window=1000)
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
    np.random.seed(42)
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/output_100m_bkup/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/",
    #     "simulation_cylinder_wscatter_wtof_blur_in_coins",
    #     True,
    #     True,
    #     1000
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/output_100m_bkup/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/",
    #     "simulation_cylinder_scatter_wtof_blur_in_coins",
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
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_wscatter_wtof_blur_in_coins.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/simulation_cylinder/cdf_100m/simulation_cylinder_scatter_wtof_blur_in_coins.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgpettofscf_1000ps_tof_bin_in_mm_block_base_scale_by_histogram",
    #     ""
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/trues_scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/scatters_wtof_1000ps.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "",
    #     128,
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/root_output/",
    #     "trues_scatters_wtof_wacf_wnorm_wgatetofscf_1000ps_block_base_histogram_scale_by_total_events_save_id_reverse",
    #     ""
    # )
    # modified_lm_cdf()

    # #################################################
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/output_20251027_human_10m_events_scit1/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_20251027_human_10m_events_scit1/",
    #     "20251027_human_true_scatter_10m_events_scit1",
    #     True,
    #     True,
    #     1000
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/output_20251027_human_10m_events_scit1/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_20251027_human_10m_events_scit1/",
    #     "20251027_human_scatter_10m_events_scit1",
    #     True
    # )
    # add_scf_to_tof_cdf(
    #     "/share/home/lyj/files/11panel_recon/20251027_human/20251027_t4_human_wrr_wtof_wengwin_410_610_tw1000ps.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_20251027_human_10m_events_scit1/20251027_human_true_scatter_10m_events_scit1.cdf",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251027_human_crystal_base/cdf_20251027_human_10m_events_scit1/20251027_human_scatter_10m_events_scit1.cdf",
    #     "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
    #     "/share/home/lyj/files/11panel_recon/20251028_cylinder_norm/20251028_norm_factor.raw",
    #     1500,
    #     "/share/home/lyj/files/11panel_recon/20251027_human/",
    #     "20251027_human_10m_events_hitogram_index_test_scit1",
    #     ""
    # )
#######################################
    # get_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/output_20251223_hoffman_10m_events_scit1/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/cdf_20251223_hoffman_10m_events_scit1/",
    #     "20251223_hoffman_10m_true_scatter_scit1",
    #     True,
    #     True,
    #     1000
    # )
    # get_scatter_cdf(
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/output_20251223_hoffman_10m_events_scit1/",
    #     "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/cdf_20251223_hoffman_10m_events_scit1/",
    #     "20251223_hoffman_10m_scatter_scit1",
    #     True
    # )
    add_scf_to_tof_cdf(
        "/share/home/lyj/files/11panel_recon/20251027_human/20251027_t4_human_wrr_wtof_wengwin_410_610_tw1000ps.cdf",
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/output_20251223_hoffman_10m_events_scit1/20251223_hoffman_10m_true_scatter_scit1.cdf",
        "/share/home/lyj/Downloads/gPET_to_SZBL/Example/output/20251223_hoffman_crystal_base/output_20251223_hoffman_10m_events_scit1/20251223_hoffman_10m_scatter_scit1.cdf",
        "/share/home/lyj/files/gate_script/HPBrain_72mm_radius_cylinder/HPBrain_72mm_radius_cylinder/ac_factor.raw",
        "/share/home/lyj/files/11panel_recon/20251028_cylinder_norm/20251028_norm_factor.raw",
        1500,
        "/share/home/lyj/files/11panel_recon/20251027_human/",
        "20251027_human_10m_events_1_3_scit1",
        ""
    )

