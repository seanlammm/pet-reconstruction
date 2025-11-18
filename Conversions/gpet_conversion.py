import numpy as np


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
    global_time = (global_time * 1e3).astype(np.uint32)  # convert 1e-6s(us) to 1e-9s(ps)

    castor_ids = np.zeros(panel_id.shape[0], dtype=np.uint32)
    if coin_type == 0:
        cir_castor_id = panel_id * 6 * 8 + module_id % 6 * 8 + crystal_id % 8
        cir_castor_id[cir_castor_id > 0] = 480 - cir_castor_id[cir_castor_id > 0]
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
    tof = global_time[:, 0] - global_time[:, 1]

    if not with_scatter:
        scatter_coins_flag = scatter_flag.sum(axis=1) > 0
        castor_ids = castor_ids[~scatter_coins_flag]
        tof = tof[~scatter_coins_flag]

    return np.column_stack((castor_ids, tof))


if __name__ == "__main__":
    num_of_counts = 0
    cdf_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    output_path = "/Users/seanlam/Downloads/"
    file_name = "test"
    output_cdf_path = output_path + file_name + ".cdf"
    output_cdh_path = output_path + file_name + ".cdh"

    for i in range(1):
        file_path = "/Users/seanlam/Downloads/coincidences.dat"
        coins_info = coins_to_cdf(file_path=file_path, coin_type=1, with_scatter=False)

        structured_array = np.empty(coins_info.shape[0], dtype=cdf_dtype)
        structured_array['time'] = 0
        structured_array['castor_id_1'] = coins_info[:, 0]
        structured_array['castor_id_2'] = coins_info[:, 1]
        buffer = structured_array.tobytes()
        with open(output_cdf_path, 'ab') as cdf:
            cdf.write(buffer)
        num_of_counts += coins_info.shape[0]

    with open(output_cdh_path, 'w') as file:
        print(f"Data filename: {file_name}.cdf", file=file)
        print(f"Number of events: {num_of_counts}", file=file)
        print(f"Data mode: list-mode", file=file)
        print(f"Data type: PET", file=file)
        print(f"Start time (s): 0", file=file)
        print(f"Duration (s): 1", file=file)
        print(f"Scanner name: PET_11panel_LD_module_base", file=file)
        print(f"Calibration factor: 1", file=file)
        print(f"lsotope: unknown", file=file)
        print(f"\n", file=file)


