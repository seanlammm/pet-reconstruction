import numpy as np


class ScannerOption:
    def __init__(self, scanner):
        self.scanner = scanner
        self.geometry_dict = self.get_geometry_info()
        self.crystal_num = int(self.geometry_dict["number of elements"])
        self.layers = int(self.geometry_dict["number of layers"])
        self.rsector_z = int(self.geometry_dict["number of rsectors axial"])
        self.rsector_xy = int(self.geometry_dict["number of rsectors"])
        self.modules_z = int(self.geometry_dict["number of modules axial"])
        self.modules_xy = int(self.geometry_dict["number of modules transaxial"])
        self.submodules_z = int(self.geometry_dict["number of submodules axial"])
        self.submodules_xy = int(self.geometry_dict["number of submodules transaxial"])
        self.crystals_z = int(self.geometry_dict["number of crystals axial"])
        self.crystals_xy = int(self.geometry_dict["number of crystals transaxial"])
        self.crystals_size_xy = float(self.geometry_dict["crystals size transaxial"])
        self.crystals_size_z = float(self.geometry_dict["crystals size axial"])
        self.crystals_depth = int(self.geometry_dict["crystals size depth"])
        self.crystals_position = np.fromfile("./Geometries/%s.glut" % scanner, dtype=np.float32).reshape([-1, 6])[:, :3]
        self.crystals_norm_vector = np.fromfile("./Geometries/%s.glut" % scanner, dtype=np.float32).reshape([-1, 6])[:, 3:]

        self.crystals_per_submodule = self.crystals_xy
        self.crystals_per_module = self.crystals_per_submodule * self.submodules_xy
        self.crystals_per_rsector = self.crystals_per_module * self.modules_xy
        self.crystal_per_ring = self.crystals_per_rsector * self.rsector_xy
        self.crystal_per_layer = self.crystal_per_ring * self.crystals_z * self.submodules_z * self.modules_z * self.rsector_z

        # sinogram parameters
        self.rings = self.crystals_z * self.submodules_z * self.modules_z * self.rsector_z
        self.views = self.crystal_per_ring // 2  # 一般为每圈晶体数/2
        self.bins = self.crystal_per_ring - 1  #self.submodules_xy * self.crystals_xy
        self.max_ring_diff = self.crystals_z * self.submodules_z * self.rsector_z * self.modules_z

    def get_geometry_info(self):
        geometry_dict = {}
        with open("./Geometries/%s.geom" % self.scanner, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾的空白字符
                line = line.strip()
                # 跳过空行和注释行
                if not line or line.startswith('#'):
                    continue
                # 按冒号分割键值对
                if ':' in line:
                    key, value = line.split(':', 1)  # 只分割第一个冒号
                    value = value.strip()
                    if len(value.split(",")) > 1:
                        value = float(value.strip().split(",")[0])
                    geometry_dict[key.strip()] = value  # 这里每两个参数的只取了第一个参数
        return geometry_dict

    def return_details(self):
        details = [
            "===== Scanner Options =====\n",
            "crystals_xy = {}\n".format(self.crystals_xy),
            "crystals_z = {}\n".format(self.crystals_z),
            "submodules_xy = {}\n".format(self.submodules_xy),
            "submodules_z = {}\n".format(self.submodules_z),
            "modules_xy = {}\n".format(self.modules_xy),
            "modules_z = {}\n".format(self.modules_z),
            "rsector_xy = {}\n".format(self.rsector_xy),
            "rsector_z = {}\n".format(self.rsector_z),
            "layers = {}\n".format(self.layers),
            "crystals_per_submodule = {}\n".format(self.crystals_per_submodule),
            "crystals_per_module = {}\n".format(self.crystals_per_module),
            "crystals_per_rsector = {}\n".format(self.crystals_per_rsector),
            "crystal_per_ring = {}\n".format(self.crystal_per_ring),
            "crystal_per_layer = {}\n".format(self.crystal_per_layer),
        ]
        return details