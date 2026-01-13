
import array_api_compat.cupy as xp
# import numpy as xp
import numpy as np
import parallelproj
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption
from Generals.PointSpreadFunction import PointSpreadFunction
from Generals.TOFGenerals import TOFOption
from tqdm import tqdm


class Projector:
    def __init__(self, recon_option: ReconOption, scanner_option: ScannerOption, psf_option: PointSpreadFunction, tof_option: TOFOption,  device_id):
        self.recon_option = recon_option
        self.psf_option = psf_option
        self.scanner_option = scanner_option
        self.device_id = device_id
        self.tof_option = tof_option
        xp.cuda.Device(device_id).use()

    def projection_forward_full(self, img, add_psf):
        if add_psf and self.psf_option is not None:
            img = self.psf_option.add_static_psf(img)
        forward_img = []
        for l in range(self.scanner_option.layers):
            for i in range(self.scanner_option.crystal_per_layer):
                i += l * self.scanner_option.crystal_per_layer
                current_pos_comb = self.recon_option.get_lor_location(
                    np.repeat(i, self.scanner_option.crystal_per_layer * self.scanner_option.layers),
                    np.arange(0, self.scanner_option.crystal_per_layer * self.scanner_option.layers)
                )
                fwd_img = parallelproj.joseph3d_fwd(
                    xstart=current_pos_comb[:, :3],
                    xend=current_pos_comb[:, 3:],
                    img=xp.asarray(img),
                    img_origin=self.recon_option.img_origin,
                    voxsize=self.recon_option.voxel_size
                )
                forward_img.append(parallelproj.backend.to_numpy_array(fwd_img))
        forward_img = np.concatenate(forward_img)
        # forward_img = parallelproj.joseph3d_fwd_tof_sino(xstart=pos_combination[:, :3],
        #                                                  xend=pos_combination[:, 3:],
        #                                                  img=img,
        #                                                  img_origin=recon_option.img_origin,
        #                                                  voxsize=recon_option.voxel_size,
        #                                                  tofbin_width=tof_option.tofbin_width,
        #                                                  sigma_tof=tof_option.sigma_tof,
        #                                                  tofcenter_offset=tof_option.tofcenter_offset,
        #                                                  nsigmas=tof_option.nsigmas,
        #                                                  ntofbins=tof_option.num_tof_bins)
        return forward_img

    def projection_forward_lors(self, img, start_index, end_index, add_psf):
        if add_psf and self.psf_option is not None:
            img = self.psf_option.add_static_psf(img)
        current_pos_comb = self.recon_option.get_lor_location(start_index, end_index)
        forward_img = parallelproj.joseph3d_fwd(
            xstart=xp.asarray(current_pos_comb[:, :3]),
            xend=xp.asarray(current_pos_comb[:, 3:]),
            img=xp.asarray(img),
            img_origin=xp.asarray(self.recon_option.img_origin),
            voxsize=xp.asarray(self.recon_option.voxel_size),
        )
        np.nan_to_num(forward_img, copy=False, nan=0, posinf=0, neginf=0)
        return parallelproj.backend.to_numpy_array(forward_img.astype(xp.float32))

    def projection_backward(self, start_index, end_index, counts, add_psf=False):  # sinogram = [id1, id2, counts]
        current_pos_comb = self.recon_option.get_lor_location(start_index, end_index)
        backward_img = parallelproj.joseph3d_back(
            xstart=xp.asarray(current_pos_comb[:, :3]),
            xend=xp.asarray(current_pos_comb[:, 3:]),
            img_fwd=xp.asarray(counts),
            img_origin=xp.asarray(self.recon_option.img_origin),
            img_shape=self.recon_option.img_dim,
            voxsize=xp.asarray(self.recon_option.voxel_size),
        )
        backward_img = parallelproj.backend.to_numpy_array(backward_img.astype(xp.float32))
        np.nan_to_num(backward_img, copy=False, nan=0, posinf=0, neginf=0)
        if add_psf and self.psf_option is not None:
            backward_img = self.psf_option.add_static_psf(backward_img)
        return backward_img

    def projection_forward_lors_wtof(self, img, lor_start_index, lor_end_index, add_psf):
        if add_psf and self.psf_option is not None:
            img = self.psf_option.add_static_psf(img)
        current_pos_comb = self.recon_option.get_lor_location(lor_start_index, lor_end_index)
        forward_img = parallelproj.joseph3d_fwd_tof_sino(
            xstart=xp.asarray(current_pos_comb[:, :3]),
            xend=xp.asarray(current_pos_comb[:, 3:]),
            img=xp.asarray(img),
            img_origin=xp.asarray(self.recon_option.img_origin),
            voxsize=xp.asarray(self.recon_option.voxel_size),
            tofbin_width=self.tof_option.tof_bin_width,
            sigma_tof=xp.asarray(self.tof_option.sigma_tof),
            tofcenter_offset=xp.asarray(self.tof_option.tof_center_offset),
            nsigmas=self.tof_option.nsigmas,
            ntofbins=self.tof_option.tof_bin_num,
        )
        np.nan_to_num(forward_img, copy=False, nan=0, posinf=0, neginf=0)
        return parallelproj.backend.to_numpy_array(forward_img.astype(xp.float32))

    def projection_backward_wtof(self, start_index, end_index, sinogram, add_psf=False):
        current_pos_comb = self.recon_option.get_lor_location(start_index, end_index)
        backward_img = parallelproj.joseph3d_back_tof_sino(
            xstart=xp.asarray(current_pos_comb[:, :3]),
            xend=xp.asarray(current_pos_comb[:, 3:]),
            img_origin=xp.asarray(self.recon_option.img_origin),
            img_shape=self.recon_option.img_dim,
            voxsize=xp.asarray(self.recon_option.voxel_size),
            img_fwd=xp.asarray(sinogram),
            tofbin_width=self.tof_option.tof_bin_width,
            sigma_tof=xp.asarray(self.tof_option.sigma_tof),
            tofcenter_offset=xp.asarray(self.tof_option.tof_center_offset),
            nsigmas=self.tof_option.nsigmas,
            ntofbins=self.tof_option.tof_bin_num,
        )
        backward_img = parallelproj.backend.to_numpy_array(backward_img.astype(xp.float32))
        np.nan_to_num(backward_img, copy=False, nan=0, posinf=0, neginf=0)
        if add_psf and self.psf_option is not None:
            backward_img = self.psf_option.add_static_psf(backward_img)
        return backward_img

    