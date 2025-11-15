import array_api_compat.cupy as xp
import numpy as np
from Applications.AppGeneral import AppGeneral
from Methods.Projector import Projector
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption


class LacToAttenuationCorrectionFactor(AppGeneral):
    def __init__(self, scanner_option, recon_option, attnenuation_image, device_id):
        super().__init__(scanner_option, recon_option)
        self.atn_img = attnenuation_image
        self.projector = Projector(recon_option, scanner_option, device_id=device_id, psf_option=None)
        self.scanner_option = scanner_option
        self.recon_option = recon_option

    def run(self):
        acf = []
        for i in range(self.scanner_option.crystal_per_layer):
            acf.append(np.column_stack((
                np.repeat(i, self.scanner_option.crystal_per_layer - i),
                np.arange(i, self.scanner_option.crystal_per_layer),
                np.repeat(1, self.scanner_option.crystal_per_layer - i)
            )))
        acf = np.concatenate(acf)
        # lin_int_proj: [id1, id2, counts, phi]
        lin_int_proj = np.fromfile(self.recon_option.output_paths["lin_int_proj_output_path"], dtype=np.float32).reshape(-1, 4)
        index_lac_to_acf = (self.scanner_option.crystal_per_layer + (self.scanner_option.crystal_per_layer - lin_int_proj[:, 0])) * lin_int_proj[:, 0] / 2 + (lin_int_proj[:, 1] - lin_int_proj[:, 0] - 1)

        acf[index_lac_to_acf, 2] = np.exp(lin_int_proj[:, 3])
        return acf


if "__name__" == "__main":
    pass