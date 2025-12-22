import numpy as np

class TOFOption:
    def __init__(self, tof_resolution, tof_bin_num, tof_bin_width_in_ps, nsigmas=3, tof_center_offset=0):
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
        self.lin_int_proj = None
        self.light_speed = 299792458  # unit: m/s
        self.tof_bin_width_in_ps = tof_bin_width_in_ps
        self.tof_bin_width_in_mm = tof_bin_width_in_ps * 1e-12 * self.light_speed * 1e3
        self.sigma_tof = np.asarray([(tof_resolution * 1e-12 * self.light_speed * 1e3) / (2 * np.sqrt(2*np.log(2)))], dtype=np.float32)