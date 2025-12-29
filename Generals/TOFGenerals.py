import numpy as np

class TOFOption:
    def __init__(self, tof_resolution, tof_bin_num, tof_range_in_ps, nsigmas=3, tof_center_offset=0):
        '''
        Args:
            tof_resolution: unit: ps
            tof_bin_num:
            tof_bin_num_width: unit: ps
            nsigmas: default 3
            tof_center_offset: default 0
        '''
        self.tof_bin_num = tof_bin_num
        self.nsigmas = nsigmas
        self.tof_center_offset = np.asarray([tof_center_offset], dtype=np.float32)
        self.light_speed = 299792458 * 1e3 / 1e12  # unit: mm/ps
        self.tof_range_in_mm = tof_range_in_ps * self.light_speed
        self.tof_bin_edges = np.linspace(-1, 1, tof_bin_num+1) * self.tof_range_in_mm / 2
        self.tof_bin_width = self.tof_bin_edges[1] - self.tof_bin_edges[0]
        self.tof_resolution = tof_resolution
        self.sigma_tof = np.asarray([(tof_resolution * self.light_speed / 2) / (2 * np.sqrt(2*np.log(2)))], dtype=np.float32)

    def get_tof_bin_index(self, tof_in_ps):
        tof_in_mm = tof_in_ps * self.light_speed / 2
        rm_option = (tof_in_mm < self.tof_bin_edges[0]) | (tof_in_mm > self.tof_bin_edges[-1])
        tof_bin_index = np.digitize(tof_in_mm, self.tof_bin_edges) - 1
        tof_bin_index[tof_bin_index < 0] = 0
        tof_bin_index[tof_bin_index >= self.tof_bin_num] = self.tof_bin_num - 1

        return tof_bin_index, rm_option
