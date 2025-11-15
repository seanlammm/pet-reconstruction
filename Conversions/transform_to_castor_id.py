import numpy as np
from Generals.ScannerGenerals import ScannerOption


def transform_id_castor(scanner_option: ScannerOption, crystal_id, submodule_id, module_id, rsector_id, layer_id):
    rsector_id_z = rsector_id % scanner_option.rsector_z
    rsector_id_xy = (rsector_id / scanner_option.rsector_z).astype(int)

    ring_id = rsector_id_z * scanner_option.modules_z * scanner_option.submodules_z * scanner_option.crystals_z \
              + (module_id / scanner_option.modules_xy).astype(int) * scanner_option.submodules_z * scanner_option.crystals_z \
              + (submodule_id / scanner_option.submodules_xy).astype(int) * scanner_option.crystals_z \
              + (crystal_id / scanner_option.crystals_xy).astype(int)

    castor_id = layer_id * scanner_option.crystal_per_layer + scanner_option.crystal_per_ring * ring_id \
                + scanner_option.crystals_per_rsector * rsector_id_xy \
                + scanner_option.crystals_per_module * (module_id % scanner_option.modules_xy) \
                + scanner_option.crystals_per_submodule * (submodule_id % scanner_option.submodules_xy) \
                + crystal_id % scanner_option.crystals_xy

    return castor_id

# def transform_id_ld(layer_id, crystal_id, submodule_id, module_id, rsector_id):
#     [crystals_xy, crystals_z, submodules_xy, submodules_z, modules_xy, modules_z, rsector_xy, rsector_z,
#      crystals_per_submodule, crystals_per_module, crystals_per_rsector, crystal_per_ring,
#      crystal_per_layer] = brain_wear_geom()
#
#     castor_id = int(crystal_id / crystals_xy) * crystals_xy * submodules_xy * rsector_xy \
#                 + module_id * crystals_xy * submodules_xy * rsector_xy * crystals_z \
#                 + (rsector_xy - rsector_id) % rsector_xy * crystals_xy * submodules_xy \
#                 + submodule_id * crystals_xy \
#                 + crystal_id % crystals_xy
#
#     return castor_id
#
#
# def transform_id_zzj(layer_id, crystal_id, submodule_id, module_id, rsector_id):
#     crystal_num_inner = 6
#     rsector_num = 32
#     submodule_num = 6
#
#     crystal_per_submodule_inner = 0
#     crystal_all_inner = 0
#     crystal_per_ring_inner = 0
#
#     idcrystal = crystal_id
#     idresctor = 2 * rsector_id + submodule_id
#     idring = module_id
#
#     castor_id = (idcrystal / crystal_num_inner + idring * crystal_num_inner) * crystal_per_ring_inner + \
#                 ((rsector_num - idresctor) % rsector_num) * crystal_num_inner + \
#                 (crystal_num_inner - 1 - idcrystal % crystal_num_inner)
#
#     return castor_id


# def transform_id_hp_brain_pet(layer_id, crystal_id, submodule_id, module_id, rsector_id):
#     crystal_num_inner_tr = 8
#     crystal_num_inner_ax = 10
#     rsector_num = 10
#     module_num = 4
#     submodule_num = 18
#     submodule_num_tr = 6
#     submodule_num_ax = 3
#
#     crystal_per_submodule_inner = crystal_num_inner_tr * crystal_num_inner_ax
#     crystal_per_module = crystal_per_submodule_inner * submodule_num
#     crystal_all_inner = crystal_per_module * rsector_num * module_num
#     crystal_per_ring_inner = crystal_num_inner_tr * rsector_num * submodule_num_tr
#
#     ring_id = submodule_num_ax * module_id + submodule_id // submodule_num_tr
#
#     castor_id = (crystal_id // crystal_num_inner_tr + ring_id * crystal_num_inner_ax) * crystal_per_ring_inner + \
#                 ((rsector_num - rsector_id) % rsector_num) * crystal_num_inner_tr * submodule_num_tr + \
#                 (submodule_num_tr - 1 - submodule_id % submodule_num_tr) * crystal_num_inner_tr + \
#                 (crystal_num_inner_tr - 1 - crystal_id % crystal_num_inner_tr)
#
#     if castor_id >= crystal_all_inner:
#         print(rsector_id)
#         print(module_id)
#         print(submodule_id)
#         print(crystal_id)
#
#         print(castor_id)
#
#         print("transform error!!")
#
#     return castor_id
