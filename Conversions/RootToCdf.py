import uproot
import numpy as np
import os
import struct
from tqdm import tqdm
import sys

sys.path.append(r"/share/home/lyj/files/git-project/pet-reconstuction/")
from Generals.ScannerGenerals import ScannerOption
from Conversions.transform_to_castor_id import transform_id_castor


class RootToCdf:
    def __init__(self, scanner_option: ScannerOption, save_path, emission_scan_root_path=None, transssion_scan_root_path=None, blank_scan_root_path=None, with_random=False, with_scatter=False):
        self.scanner_option = scanner_option
        self.ex_path = emission_scan_root_path
        self.tx_path = transssion_scan_root_path
        self.bx_path = blank_scan_root_path
        self.scan_type = None
        self.save_path = save_path
        self.w_random = with_random
        self.w_scatter = with_scatter
        self.num_of_event = 0
        self.time = 0
        self.singles_counts = None
        self.delayed_rates = None

        self.path_check()

    def path_check(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        if self.ex_path is None:
            if self.bx_path is None or self.tx_path is None:
                print("No enough file path input.")
            else:
                self.scan_type = "transmission_scan"
        else:
            self.scan_type = "emission_scan"

        # 检查cdf文件是否存在(放在写文件部分)

    def get_root_file_name(self):
        if self.scan_type == "transmission_scan":
            tx_root_files = [i for i in os.listdir(self.tx_path) if ".root" in i]
            bx_root_files = [i for i in os.listdir(self.bx_path) if ".root" in i]
            return None, tx_root_files, bx_root_files
        elif self.scan_type == "emission_scan":
            ex_root_files = [i for i in os.listdir(self.ex_path) if ".root" in i]
            self.num_of_event = len(ex_root_files)
            return ex_root_files, None, None
        else:
            return -1

    def get_castor_id(self, rsector_id, module_id, submodule_id, crystal_id):
        castor_ids = transform_id_castor(self.scanner_option, crystal_id, submodule_id, module_id, rsector_id, layer_id=0)
        coins_len = int(castor_ids.shape[0] / 2)
        castor_ids = np.column_stack((castor_ids[:coins_len], castor_ids[coins_len:]))
        return castor_ids

    # def source_split_by_tof(self, time):
    #     # 条件 1：根据与中心的距离判断
    #     tof = time[:, 1] - time[:, 0]  # ps --> s
    #     dist_to_center = 3e8 * tof * 1000  # mm
    #     tof_split_flag = dist_to_center > 210  # split_flag==True: Tx, split_flag==False: Ex,
    #     # 条件 2：根据旋转源的实时位置判断
    #     current_time = np.floor(time[0][0])  # 一个 root 文件里的位置都是一样的
    #
    #
    #     return split_flag

    def ground_true_source_check(self, source_id, source_pos):
        unique_source = np.unique(source_id[:, 0])
        if self.scan_type == "emission_scan":
            return unique_source, None
        elif unique_source.shape[0] == 1:
            if self.scan_type == "transmission_scan":
                return None, unique_source
        elif unique_source.shape[0] >= 2:
            source_pos_x = source_pos[:, 0]
            source_pos_y = source_pos[:, 1]

            # 根据距离区别源，将距离中心最远的源视作Tx
            source_distance_to_center = []
            for i in range(unique_source.shape[0]):
                source_event_index = np.where(source_id[:, 0] == unique_source[i])[0]
                source_distance_to_center.append(np.mean(source_pos_x[source_event_index] ** 2 + source_pos_y[source_event_index] ** 2))

            tx_source = unique_source[np.argmax(np.array(source_distance_to_center))]
            ex_source = np.delete(unique_source, np.argmax(np.array(source_distance_to_center)))

            return ex_source, tx_source
        else:
            print("More than 2 source...")
            return -1

    def read_root(self, root_file_path):
        with uproot.open(root_file_path) as file:
            coins_tree = file["Coincidences"]
            # 确认事件数
            source_id = np.column_stack((coins_tree["sourceID1"].array(library="numpy"), coins_tree["sourceID2"].array(library="numpy")))
            compton_phantom = np.column_stack((coins_tree["comptonPhantom1"].array(library="numpy"), coins_tree["comptonPhantom2"].array(library="numpy")))
            rayleigh_phantom = np.column_stack((coins_tree["RayleighPhantom1"].array(library="numpy"), coins_tree["RayleighPhantom2"].array(library="numpy")))
            compton_crystal = np.column_stack((coins_tree["comptonCrystal1"].array(library="numpy"), coins_tree["comptonCrystal2"].array(library="numpy")))
            rayleigh_crystal = np.column_stack((coins_tree["RayleighCrystal1"].array(library="numpy"), coins_tree["RayleighCrystal2"].array(library="numpy")))

            time = np.column_stack((coins_tree["time1"].array(library="numpy"), coins_tree["time2"].array(library="numpy")))
            source_pos = np.column_stack((coins_tree["sourcePosX1"].array(library="numpy"), coins_tree["sourcePosY1"].array(library="numpy")))
            rsector_id = np.concatenate((coins_tree["rsectorID1"].array(library="numpy"), coins_tree["rsectorID2"].array(library="numpy")))
            module_id = np.concatenate((coins_tree["moduleID1"].array(library="numpy"), coins_tree["moduleID2"].array(library="numpy")))
            submodule_id = np.concatenate((coins_tree["submoduleID1"].array(library="numpy"), coins_tree["submoduleID2"].array(library="numpy")))
            crystal_id = np.concatenate((coins_tree["crystalID1"].array(library="numpy"), coins_tree["crystalID2"].array(library="numpy")))

        # 确认有效事件（有效 --> 需要）
        num_of_events = source_id.shape[0]
        effective_index = np.ones(num_of_events, dtype=bool)
        if not self.w_random:
            effective_index[source_id[:, 0] != source_id[:, 1]] = 0
        if not self.w_scatter:
            compton_phantom_index = (compton_phantom > 0).all(axis=1)
            rayleigh_phantom_index = (rayleigh_phantom > 0).all(axis=1)
            compton_crystal_index = (compton_crystal > 0).all(axis=1)
            rayleigh_crystal_index = (rayleigh_crystal > 0).all(axis=1)
            effective_index[compton_phantom_index | rayleigh_phantom_index | compton_crystal_index | rayleigh_crystal_index] = 0

        rsector_id = rsector_id[np.tile(effective_index, 2)]
        module_id = module_id[np.tile(effective_index, 2)]
        submodule_id = submodule_id[np.tile(effective_index, 2)]
        crystal_id = crystal_id[np.tile(effective_index, 2)]
        time = time[effective_index, :]
        source_id = source_id[effective_index, :]
        source_pos = source_pos[effective_index, :]

        # 计算 castor 编号
        castor_ids = self.get_castor_id(rsector_id, module_id, submodule_id, crystal_id)
        # 事件读出 (time, id_1, id_2, tof)
        tof = (time[:, 1] - time[:, 0]) * 1e9  # s --> ps
        coins = np.column_stack((time[:, 0], castor_ids, tof))

        ex_source_id, tx_source_id = self.ground_true_source_check(source_id, source_pos)
        # 随机事件（ex--tx）按照 source_id_1 分类
        ex_coins = None
        tx_coins = None
        if ex_source_id is not None:
            ex_coins = coins[np.isin(source_id[:, 0], ex_source_id) & np.isin(source_id[:, 1], ex_source_id)]
        if tx_source_id is not None:
            tx_coins = coins[np.isin(source_id[:, 0], tx_source_id) & np.isin(source_id[:, 1], tx_source_id)]

        return ex_coins, tx_coins

    def write_cdf(self, coins_data, with_tof, file_name, add_mode: bool):
        cdf_path = os.path.join(self.save_path, "%s.cdf" % file_name)
        if not add_mode and os.path.exists(cdf_path):
            raise Exception("需要手动删除已存在的输出文件")

        if with_tof:
            int_data = coins_data[:, :3].astype(int)
            float32_data = coins_data[:, 3].astype(np.float32).reshape(-1, 1)

            current_dtype = [('time', 'i4'), ('tof', 'f4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]  # 32位整数
            structured_array = np.empty(coins_data.shape[0], dtype=current_dtype)
            structured_array['time'] = int_data[:, 0]
            structured_array['tof'] = float32_data[:, 0]
            structured_array['castor_id_1'] = int_data[:, 1]
            structured_array['castor_id_2'] = int_data[:, 2]
        else:
            coins_data = coins_data[:, :3].astype(int)
            current_dtype = [('time', 'i4'), ('castor_id_1', 'i4'), ('castor_id_2', 'i4')]  # 32位整数
            structured_array = np.empty(coins_data.shape[0], dtype=current_dtype)
            structured_array['time'] = coins_data[:, 0]
            structured_array['castor_id_1'] = coins_data[:, 1]
            structured_array['castor_id_2'] = coins_data[:, 2]

        self.num_of_event += coins_data.shape[0]
        buffer = structured_array.tobytes()
        with open(cdf_path, 'ab') as cdf:
            cdf.write(buffer)

    def run(self):
        if self.scan_type == "emission_scan":
            emission_root_files, _, _ = self.get_root_file_name()
            file_name = "emission"
            if self.w_random:
                file_name = "%s_wrandom" % file_name
            if self.w_scatter:
                file_name = "%s_wscatter" % file_name
            for f in tqdm(emission_root_files, desc="Writing emission cdf..."):
                root_file_path = os.path.join(self.ex_path, f)
                ex_coins, _ = self.read_root(root_file_path)
                self.write_cdf(ex_coins, False, file_name, (f != emission_root_files[0]))
        else:
            _, tx_root_files, bx_root_files = self.get_root_file_name()

            for f in tqdm(tx_root_files, desc="Writing object transmission scan cdf..."):
                root_file_path = os.path.join(self.tx_path, f)
                ex_coins, tx_coins = self.read_root(root_file_path)
                self.write_cdf(ex_coins, False, "emission", (f != tx_root_files[0]))
                self.write_cdf(tx_coins, False, "object_tx_scan", (f != tx_root_files[0]))

            for f in tqdm(bx_root_files, desc="Writing blank transmission scan cdf..."):
                root_file_path = os.path.join(self.bx_path, f)
                _, bx_coins = self.read_root(root_file_path)
                self.write_cdf(bx_coins, False, "blank_tx_scan", (f != bx_root_files[0]))

    def get_randoms_and_scatters(self):
        # 暂时默认只用于纯 emission 扫描
        emission_root_files, _, _ = self.get_root_file_name()
        for f in tqdm(emission_root_files, desc="Writing emission cdf..."):
            root_file_path = os.path.join(self.ex_path, f)
            randoms_coins, scatters_coins = self.get_coins_in_randoms_and_scatters(root_file_path)
            self.write_cdf(randoms_coins, False, "randoms", (f != emission_root_files[0]))
            self.write_cdf(scatters_coins, False, "scatters", (f != emission_root_files[0]))

    def get_coins_in_randoms_and_scatters(self, root_file_path):
        with uproot.open(root_file_path) as file:
            coins_tree = file["Coincidences"]
            # 确认事件数
            source_id = np.column_stack((coins_tree["sourceID1"].array(library="numpy"), coins_tree["sourceID2"].array(library="numpy")))
            compton_phantom = np.column_stack((coins_tree["comptonPhantom1"].array(library="numpy"), coins_tree["comptonPhantom2"].array(library="numpy")))
            rayleigh_phantom = np.column_stack((coins_tree["RayleighPhantom1"].array(library="numpy"), coins_tree["RayleighPhantom2"].array(library="numpy")))
            compton_crystal = np.column_stack((coins_tree["comptonCrystal1"].array(library="numpy"), coins_tree["comptonCrystal2"].array(library="numpy")))
            rayleigh_crystal = np.column_stack((coins_tree["RayleighCrystal1"].array(library="numpy"), coins_tree["RayleighCrystal2"].array(library="numpy")))

            time = np.column_stack((coins_tree["time1"].array(library="numpy"), coins_tree["time2"].array(library="numpy")))
            rsector_id = np.concatenate((coins_tree["rsectorID1"].array(library="numpy"), coins_tree["rsectorID2"].array(library="numpy")))
            module_id = np.concatenate((coins_tree["moduleID1"].array(library="numpy"), coins_tree["moduleID2"].array(library="numpy")))
            submodule_id = np.concatenate((coins_tree["submoduleID1"].array(library="numpy"), coins_tree["submoduleID2"].array(library="numpy")))
            crystal_id = np.concatenate((coins_tree["crystalID1"].array(library="numpy"), coins_tree["crystalID2"].array(library="numpy")))

        # 获取随机事件与散射事件
        num_of_events = source_id.shape[0]
        randoms_index = np.zeros(num_of_events)
        scatters_index = np.zeros(num_of_events)

        randoms_index[source_id[:, 0] != source_id[:, 1]] = 1

        compton_phantom_index = (compton_phantom > 0).all(axis=1)
        rayleigh_phantom_index = (rayleigh_phantom > 0).all(axis=1)
        compton_crystal_index = (compton_crystal > 0).all(axis=1)
        rayleigh_crystal_index = (rayleigh_crystal > 0).all(axis=1)
        scatters_index[compton_phantom_index | rayleigh_phantom_index | compton_crystal_index | rayleigh_crystal_index] = 1

        # 计算 castor 编号
        castor_ids = self.get_castor_id(rsector_id, module_id, submodule_id, crystal_id)
        # 事件读出 (time, id_1, id_2, tof)
        tof = (time[:, 1] - time[:, 0]) * 1e9  # s --> ps

        randoms_coins = np.column_stack((time[:, 0], castor_ids, tof))[randoms_index, :]
        scatters_coins = np.column_stack((time[:, 0], castor_ids, tof))[scatters_index, :]
        return randoms_coins, scatters_coins

    def stack_up_singles_counts(self, write_out=False):
        # 暂时默认只用于纯 emission 扫描
        self.singles_counts = np.zeros(self.scanner_option.crystal_per_layer, dtype=np.uint32)
        emission_root_files, _, _ = self.get_root_file_name()
        for f in tqdm(emission_root_files, desc="Stacking singles counts..."):
            root_file_path = os.path.join(self.ex_path, f)
            self.singles_counts += self.get_singles_counts(root_file_path).astype(np.uint32)
        if write_out:
            self.singles_counts.astype(np.uint32).tofile(os.path.join(self.save_path, "singles_counts.raw"))

    def get_singles_counts(self, root_file_path):
        with uproot.open(root_file_path) as file:
            singles_tree = file["Singles"]
            rsector_id = singles_tree["rsectorID"].array(library="numpy")
            module_id = singles_tree["moduleID"].array(library="numpy")
            submodule_id = singles_tree["submoduleID"].array(library="numpy")
            crystal_id = singles_tree["crystalID"].array(library="numpy")
        castor_ids = transform_id_castor(self.scanner_option, crystal_id, submodule_id, module_id, rsector_id)
        uni_id, counts = np.unique(castor_ids, return_counts=True)
        if counts.shape[0] != self.scanner_option.crystal_per_layer:
            singles_counts = np.zeros(self.scanner_option.crystal_per_layer, dtype=np.uint32)
            singles_counts[uni_id] = counts
            return singles_counts
        return counts

    def get_singles_random_rate(self):
        self.stack_up_singles_counts(write_out=False)
        singles_rate = self.singles_counts / 100
        singles_random_rate = np.zeros((self.scanner_option.crystal_per_layer+1)*self.scanner_option.crystal_per_layer / 2)
        for i in tqdm(range(self.scanner_option.crystal_per_layer)):
            id_0 = np.ones(self.scanner_option.crystal_per_layer - i)
            id_1 = np.arange(i, self.scanner_option.crystal_per_layer - i)

            coin_time_window = 5 * 1e-9  # time_window * seconds per time window
            singles_rate_0 = singles_rate[id_0]
            singles_rate_1 = singles_rate[id_1]
            randoms_rate = 2 * coin_time_window * singles_rate_0 * singles_rate_1
            index = ((self.scanner_option.crystal_per_layer + self.scanner_option.crystal_per_layer - id_0 + 1) * id_0 / 2 + id_1 - id_0).astype(int)
            singles_random_rate[index] = randoms_rate

        singles_random_rate.astype(np.float32).tofile(os.path.join(self.save_path, "singles_random_rate.raw"))

    def get_delayed_coins(self, root_file_path):
        with uproot.open(root_file_path) as file:
            delayed_tree = file["delay"]
            rsector_id = np.concatenate((delayed_tree["rsectorID1"].array(library="numpy"), delayed_tree["rsectorID2"].array(library="numpy")))
            module_id = np.concatenate((delayed_tree["moduleID1"].array(library="numpy"), delayed_tree["moduleID2"].array(library="numpy")))
            submodule_id = np.concatenate((delayed_tree["submoduleID1"].array(library="numpy"), delayed_tree["submoduleID2"].array(library="numpy")))
            crystal_id = np.concatenate((delayed_tree["crystalID1"].array(library="numpy"), delayed_tree["crystalID2"].array(library="numpy")))

            # 计算 castor 编号
            castor_ids = self.get_castor_id(rsector_id, module_id, submodule_id, crystal_id)
            swap_flag = castor_ids[:, 0] > castor_ids[:, 1]
            swap_value = castor_ids[swap_flag, 0]
            castor_ids[swap_flag, 0] = castor_ids[swap_flag, 1]
            castor_ids[swap_flag, 1] = swap_value

            uni_ids, counts = np.unique(castor_ids, axis=0, return_counts=True)
            index = ((self.scanner_option.crystal_per_layer + self.scanner_option.crystal_per_layer - uni_ids[:, 0] + 1) * uni_ids[:, 0] / 2 + uni_ids[:, 1] - uni_ids[:, 0]).astype(int)
            self.delayed_rates[index] += counts

    def get_delayed_random_rate(self, write_out):
        # 暂时默认只用于纯 emission 扫描
        self.delayed_rates = np.zeros(int((self.scanner_option.crystal_per_layer+1)*self.scanner_option.crystal_per_layer / 2), dtype=np.float32)
        emission_root_files, _, _ = self.get_root_file_name()
        for f in tqdm(emission_root_files, desc="Stacking randoms counts..."):
            root_file_path = os.path.join(self.ex_path, f)
            self.get_delayed_coins(root_file_path)
        self.delayed_rates /= 100
        if write_out:
            self.delayed_rates.astype(np.float32).tofile(os.path.join(self.save_path, "delayed_random_rates.raw"))


if __name__ == "__main__":
    os.chdir(r"D:\linyuejie\git-project\pet-reconstuction")
    scanner_option = ScannerOption("cardiac_lyj")
    details = scanner_option.return_details()

    root2cdf = RootToCdf(
        scanner_option=scanner_option,
        save_path=r"E:\simulation\cardiac_simulation",
        emission_scan_root_path=r"E:\simulation\cardiac_simulation\output",
        #transssion_scan_root_path=r"G:\cardiac_tx_simulation\cardiac_iq_phantom_tx_360s_1mci\output",
        #blank_scan_root_path=r"G:\cardiac_tx_simulation\cardiac_iq_phantom_blank_360s_1mci\output",
        with_random=True,
        with_scatter=False
    )
    root2cdf.get_delayed_random_rate(write_out=True)
