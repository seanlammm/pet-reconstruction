# import array_api_compat.cupy as xp
# import numpy as np
# from Applications.AppGeneral import AppGeneral
# from Methods.Projector import Projector
# from Generals.ReconGenerals import ReconOption
# from Generals.ScannerGenerals import ScannerOption
# from tqdm import tqdm
#
#
# class MumapToAttenuationCorrectionFactor(AppGeneral):
#     def __init__(self, scanner_option, recon_option, attnenuation_image, device_id):
#         super().__init__(scanner_option, recon_option)
#         self.atn_img = attnenuation_image
#         self.projector = Projector(recon_option, scanner_option, device_id=device_id, psf_option=None)
#         self.scanner_option = scanner_option
#         self.recon_option = recon_option
#
#     def run(self):
#         ids = []
#         for i in tqdm(range(self.scanner_option.crystal_per_layer)):
#             ids.append(np.column_stack((
#                 np.repeat(i, self.scanner_option.crystal_per_layer - i),
#                 np.arange(i, self.scanner_option.crystal_per_layer),
#             )))
#         ids = np.concatenate(ids)
#         attenuation_factor = self.projector.projection_forward_full(xp.asarray(self.atn_img), add_psf=False)
#         attenuation_factor = np.exp(attenuation_factor)
#         # acf = np.column_stack((acf, attenuation_factor))
#
#         return attenuation_factor
#
#
# if __name__ == "__main__":
#     import os
#     os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
#     device_id = 2
#     scanner_option = ScannerOption("PET_11panel_LD")
#
#     recon_option = ReconOption(
#         img_dim=np.array([300, 300, 300]),
#         voxel_size=np.array([1, 1, 1]),
#         output_dir="",
#         ex_cdf_path="",
#         tx_cdf_path="",
#         bx_cdf_path="",
#         num_of_subsets=1,
#         num_of_iterations=1,
#         psf_option=None,
#         scanner_option=scanner_option,
#         device_id=device_id
#     )
#
#     atn_img = np.fromfile(r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/registration_mumap.img", dtype=np.float32).reshape([300, 300, 300]).transpose([2, 1, 0])
#     # atn_img /= 10
#     get_acf = MumapToAttenuationCorrectionFactor(scanner_option, recon_option, atn_img, device_id)
#     acf = get_acf.run()
#     acf.astype(np.float32).tofile(r"/share/home/lyj/files/11panel_recon/20250227_hoffman/t14_old_wrong_energy/ac_factor_transpose_210.raw")

import array_api_compat.cupy as xp
import numpy as np
from Applications.AppGeneral import AppGeneral
from Methods.Projector import Projector
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption
from tqdm import tqdm


class MumapToAttenuationCorrectionFactor(AppGeneral):
    def __init__(self, scanner_option, recon_option, attnenuation_image, device_id):
        super().__init__(scanner_option, recon_option)
        self.atn_img = attnenuation_image
        self.projector = Projector(recon_option, scanner_option, device_id=device_id, psf_option=None)
        self.scanner_option = scanner_option
        self.recon_option = recon_option

    def run(self):
        self.atn_img /= 10
        self.atn_img = self.atn_img.transpose([2, 1, 0])
        attenuation_factor = self.projector.projection_forward_full(xp.asarray(self.atn_img), add_psf=False)
        attenuation_factor = np.exp(attenuation_factor)

        return attenuation_factor


if __name__ == "__main__":
    import os
    os.chdir(r"/share/home/lyj/files/git-project/pet-reconstuction")
    device_id = 2
    scanner_option = ScannerOption("PET_11panel_LD")

    recon_option = ReconOption(
        img_dim=np.array([300, 300, 300]),
        voxel_size=np.array([1, 1, 1]),
        output_dir="",
        ex_cdf_path="",
        tx_cdf_path="",
        bx_cdf_path="",
        num_of_subsets=1,
        num_of_iterations=1,
        psf_option=None,
        scanner_option=scanner_option,
        device_id=device_id
    )
    # 只需要保证 mumap 是与重建图像方向一致，且单位为 cm-1
    atn_img = np.fromfile(r"/share/home/lyj/files/11panel_recon/20250521_medium_hoffman/t6_medium_hoffman_20250521.img", dtype=np.float32).reshape([300, 300, 300])
    get_acf = MumapToAttenuationCorrectionFactor(scanner_option, recon_option, atn_img, device_id)
    acf = get_acf.run()
    acf.astype(np.float32).tofile(r"/share/home/lyj/files/11panel_recon/20250521_medium_hoffman/acfactor_medium_hoffman_20250521.raw")