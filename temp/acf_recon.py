from Methods.Projector import Projector
from Generals.ScannerGenerals import ScannerOption
from Generals.ReconGenerals import ReconOption
import numpy as np

scanner_option = ScannerOption("PET_11panel_LD")
recon_option = ReconOption(
    img_dim=np.array([300, 300, 300]),
    voxel_size=np.array([1, 1, 1]),
    ex_cdf_path=None, tx_cdf_path=None, bx_cdf_path=None,
    output_dir=r"path/to/output",
    scanner_option=scanner_option,
    psf_option=None,
    num_of_iterations=1,
    num_of_subsets=4,
    device_id=2)

projector = Projector(scanner_option=scanner_option, recon_option=recon_option, psf_option=None, device_id=2)
ids = []
for i in range(57600):
    ids.append(np.column_stack((
        np.repeat(i, 57600),
        np.arange(0, 57600),
    )))
ids = np.concatenate(ids)

ac_factor = np.fromfile(r"path/to/ac_factor", dtype=np.float32)
ac_factor = np.exp(ac_factor)
sinogram = np.column_stack((ids, ac_factor))
del ac_factor
del ids
recon_mumap = projector.projection_backward(sinogram, add_psf=False)
recon_mumap.astype(np.float32).tofile("path/to/output")

