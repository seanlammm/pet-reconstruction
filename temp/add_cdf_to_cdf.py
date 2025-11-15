import numpy as np
import os
from Generals.ScannerGenerals import ScannerOption
from tqdm import tqdm


def add_acf_to_cdf():
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    ac_factor = np.fromfile(
        r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/medium_hoffman_cylinder_ac_factor.raw", dtype=np.float32)
    ip_type = [('time', 'i4'), ('random_rate', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('random_rate', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    cdf_data = np.fromfile(
        r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/hoffman_wrr.cdf",
        dtype=ip_type)
    op_path = r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/hoffman_wrr_w210acf.cdf"
    id_1 = cdf_data["castor_id_1"]
    id_2 = cdf_data["castor_id_2"]
    random_rate = cdf_data["random_rate"]
    del cdf_data
    index = ((scanner_option.crystal_per_layer + scanner_option.crystal_per_layer - id_1 + 1) * id_1 / 2 + id_2 - id_1).astype(int)
    acf = ac_factor[index]
    structured_array = np.empty(id_1.shape[0], dtype=op_type)
    structured_array['time'] = 0
    structured_array['castor_id_1'] = id_1
    structured_array['castor_id_2'] = id_2
    structured_array['random_rate'] = random_rate
    structured_array['acf'] = acf
    # structured_array['norm'] = np.ones(acf.shape[0])
    buffer = structured_array.tobytes()
    with open(op_path, 'ab') as cdf:
        cdf.write(buffer)


def generate_norm_file():
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    ac_factor = np.fromfile(r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/medium_hoffman_cylinder_ac_factor.raw", dtype=np.float32)
    op_path = r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/norm_file_mm-1.cdf"
    num_counts = 0
    for i in tqdm(range(scanner_option.crystal_per_layer)):
        ids = np.column_stack((
            np.repeat(i, scanner_option.crystal_per_layer),
            np.arange(0, scanner_option.crystal_per_layer),
        ))

        swap_flag = ids[:, 0] > ids[:, 1]
        swap_value = ids[swap_flag, 0]
        ids[swap_flag, 0] = ids[swap_flag, 1]
        ids[swap_flag, 1] = swap_value

        index = ((scanner_option.crystal_per_layer + scanner_option.crystal_per_layer - ids[:, 0] + 1) * ids[:, 0] / 2 + ids[:, 1] - ids[:, 0]).astype(int)
        acf = ac_factor[index]
        op_type = [('acf', 'f4'), ('norm', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
        structured_array = np.empty(ids.shape[0], dtype=op_type)
        structured_array['castor_id_1'] = ids[:, 0]
        structured_array['castor_id_2'] = ids[:, 1]
        structured_array['acf'] = acf
        structured_array['norm'] = np.ones(acf.shape[0])
        buffer = structured_array.tobytes()
        with open(op_path, 'ab') as cdf:
            cdf.write(buffer)
        num_counts += ids.shape[0]
    print("Num of counts: %d" % num_counts)


def add_acf_to_norm_file():
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    ac_factor = np.fromfile(r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/medium_hoffman_cylinder_ac_factor.raw", dtype=np.float32)
    norm_file_path = r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/normalization_11panel_20250321_t4t6_new.cdf"
    op_path = r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/norm_file_mm-1.cdf"

    ip_dtype = [('norm', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]

    item_size = np.dtype(ip_dtype).itemsize
    norm_file_len = os.path.getsize(norm_file_path)/item_size
    subset_size = 100

    num_counts = 0
    for i in range(subset_size):
        start_index = int(norm_file_len / 100) * i
        if i == subset_size - 1:
            end_index = int(norm_file_len / 100) * (i+1)
        else:
            end_index = norm_file_len

        counts = end_index - start_index
        offset = start_index * item_size
        nf_data = np.fromfile(norm_file_path, count=counts, offset=offset, dtype=ip_dtype)
        norm = nf_data["norm"]
        castor_id_1 = nf_data["castor_id_1"]
        castor_id_2 = nf_data["castor_id_2"]
        ids = np.column_stack((castor_id_1, castor_id_2))
        del nf_data

        swap_flag = ids[:, 0] > ids[:, 1]
        swap_value = ids[swap_flag, 0]
        ids[swap_flag, 0] = ids[swap_flag, 1]
        ids[swap_flag, 1] = swap_value

        index = ((scanner_option.crystal_per_layer + scanner_option.crystal_per_layer - ids[:, 0] + 1) * ids[:, 0] / 2 + ids[:, 1] - ids[:, 0]).astype(int)
        acf = ac_factor[index]
        op_type = [('acf', 'f4'), ('norm', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
        structured_array = np.empty(ids.shape[0], dtype=op_type)
        structured_array['castor_id_1'] = castor_id_1
        structured_array['castor_id_2'] = castor_id_2
        structured_array['acf'] = acf
        structured_array['norm'] = norm
        buffer = structured_array.tobytes()
        with open(op_path, 'ab') as cdf:
            cdf.write(buffer)
        num_counts += ids.shape[0]
    print("Num of counts: %d" % num_counts)


def add_acf_to_cdf():
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    scanner_option = ScannerOption("PET_11panel_LD")
    ac_factor = np.fromfile(
        r"/share/home/lyj/files/11panel_recon/20250328_medium_hoffman/medium_hoffman_cylinder_ac_factor.raw", dtype=np.float32)
    ip_type = [('time', 'i4'), ('random_rate', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    op_type = [('time', 'i4'), ('acf', 'f4'), ('random_rate', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]
    cdf_data = np.fromfile(
        r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/hoffman_wrr.cdf",
        dtype=ip_type)
    op_path = r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/hoffman_wrr_w210acf.cdf"
    id_1 = cdf_data["castor_id_1"]
    id_2 = cdf_data["castor_id_2"]
    random_rate = cdf_data["random_rate"]
    del cdf_data
    index = ((scanner_option.crystal_per_layer + scanner_option.crystal_per_layer - id_1 + 1) * id_1 / 2 + id_2 - id_1).astype(int)
    acf = ac_factor[index]
    structured_array = np.empty(id_1.shape[0], dtype=op_type)
    structured_array['time'] = 0
    structured_array['castor_id_1'] = id_1
    structured_array['castor_id_2'] = id_2
    structured_array['random_rate'] = random_rate
    structured_array['acf'] = acf
    # structured_array['norm'] = np.ones(acf.shape[0])
    buffer = structured_array.tobytes()
    with open(op_path, 'ab') as cdf:
        cdf.write(buffer)

if __name__ == "__main__":
    # add_acf_to_cdf()
    # generate_norm_file()
    add_acf_to_norm_file


