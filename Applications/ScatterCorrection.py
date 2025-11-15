import array_api_compat.cupy as xp
import numpy as np
from Applications.AppGeneral import AppGeneral
from Methods.Projector import Projector
from Generals.ReconGenerals import ReconOption
from Generals.ScannerGenerals import ScannerOption
from tqdm import tqdm

from OpenSourceProgram.OpenSSS.functions import CropAndDownscale, SinogramCoordinates, SinogramToSpatial, normalize_array, MaskGenerator
from OpenSourceProgram.OpenSSS.SingleScatterSimulationTOF import SingleScatterSimulationTOF


class ScatterCorrection(AppGeneral):
    def __init__(self, scanner_option, recon_option,
                 save_path, energy_resolution, tof_resolution,
                 attenuation_img, attenuation_img_size, activity_img, activity_img_size,
                 # params for simulation setting
                 desire_dimension, simulated_rings, simulated_detectors, sample_step, acceleration_factor, tof_range, energy_threshold, tof_bins):
        super().__init__(scanner_option, recon_option)
        self.attenuation_table = np.load("./OpenSourceProgram/OpenSSS/AttenuationTable.npy")
        self.save_path = save_path
        self.energy_resolution = energy_resolution
        self.tof_resolution = tof_resolution
        # # Import attenuation table and see where to find the row of current attenuation energy
        # AttenuationTable = np.load(f'{Path2Data}/AttenuationTable.npy')

        self.geometry, self.norm_vector = self.get_geometry_info()
        self.attenuation_img = attenuation_img
        self.attenuation_img_size = attenuation_img_size
        self.activity_img = activity_img
        self.activity_img_size = activity_img_size

        # Parameter for SSS simulation
        self.desire_dimension = desire_dimension  # [x, y, z]
        self.simulated_rings = simulated_rings
        self.simulated_detectors = simulated_detectors
        self.sample_step = sample_step
        self.acceleration_factor = acceleration_factor
        self.tof_range = tof_range
        self.energy_threshold = energy_threshold
        self.tof_bins = tof_bins

    def get_geometry_info(self):
        geometry = self.scanner_option.crystals_position / 10
        geometry = geometry.reshape([self.scanner_option.rings, self.scanner_option.crystal_num // self.scanner_option.rings, 3])

        norm_vector = self.scanner_option.crystals_norm_vector
        norm_vector = norm_vector.reshape([self.scanner_option.rings, self.scanner_option.crystal_num // self.scanner_option.rings, 3])
        return geometry, norm_vector

    def run(self):
        DeviceSize = [
            (np.max(self.geometry[:, :, 0]) - np.min(self.geometry[:, :, 0])) * 10,
            (np.max(self.geometry[:, :, 1]) - np.min(self.geometry[:, :, 1])) * 10,
            (np.max(self.geometry[:, :, 2]) - np.min(self.geometry[:, :, 2])) * 10
        ]

        desire_size = np.array([self.activity_img.shape[0] * self.activity_img_size[0],
                                self.activity_img.shape[0] * self.activity_img_size[1],
                                DeviceSize[2]])  # [x, y, z]

        # It is possible to crop and downscale the images. This is recommended to avoid
        # running out of memmory and crashing the computer
        # Units in mm
        DesiredSize = np.array([self.activity_img.shape[0] * self.activity_img_size[0],
                                self.activity_img.shape[0] * self.activity_img_size[1],
                                DeviceSize[2]])

        # Coordinates for the bounds of the image to be used to estimate scatters
        # in the format [xStart, yStart, zStart, xEnd, yEnd, zEnd] and in cm
        ImageSize = np.array([-self.desire_dimension[0] / 2, -self.desire_dimension[1] / 2, -self.desire_dimension[2] / 2,
                              self.desire_dimension[0] / 2, self.desire_dimension[1] / 2, self.desire_dimension[2] / 2])

        ImageSize = np.array([x / 10 for x in ImageSize])  # Convert to cm

        LORCoordinates, SinogramIndex = SinogramCoordinates(scanner_option.rsector_xy, scanner_option.rsector_z, scanner_option.modules_z, scanner_option.modules_xy, scanner_option.crystals_xy, scanner_option.crystals_z)
        DetectorCoordinates, RingCoordinates = SinogramToSpatial(scanner_option.rsector_xy, scanner_option.rsector_z, scanner_option.modules_z, scanner_option.modules_xy, scanner_option.crystals_xy, scanner_option.crystals_z, self.geometry)

        AttenuationMapDownscaled = CropAndDownscale(self.attenuation_img, self.attenuation_img_size, desire_size, self.desire_dimension,
                                                    True, 'edge', True, True, 3)

        # Very low attenuation values (such as air) do not influence the SSS significantly, so can be skipped by making them 0 (no attenuation)
        AttenuationMapDownscaled[AttenuationMapDownscaled < 0.001] = 0
        ActivityMapDownscaled = CropAndDownscale(self.activity_img, self.activity_img_size, desire_size, self.desire_dimension, True,
                                                 'edge', True, True, 3)
        # Generate Attenuation Mask & Interpolated scatters files
        print('Start generating Attenuation mask & Interpolated scatters')


        # Generate Attenuation Mask
        AttenuationMask = MaskGenerator(ActivityMapDownscaled, AttenuationMapDownscaled, ImageSize, self.geometry,
                                        LORCoordinates, SinogramIndex, True, self.acceleration_factor)
        np.save(f'{self.save_path}/AttenuationMask.npy', AttenuationMask)
        print('completed!\n')

        InterpolatedScatters = SingleScatterSimulationTOF(ActivityMapDownscaled, AttenuationMapDownscaled,
                                                          ImageSize, self.geometry, LORCoordinates, SinogramIndex,
                                                          self.norm_vector, scanner_option.crystals_size_xy, self.attenuation_table,
                                                          self.energy_resolution, self.energy_threshold, self.simulated_rings,
                                                          self.simulated_detectors, self.sample_step, self.tof_resolution,
                                                          self.tof_range, self.tof_bins, self.save_path)

        print('completed!\n')

        # save interpolated scatter for further usage
        np.save(f'{self.save_path}/Interpolated_Scatters.npy', InterpolatedScatters)


if __name__ == "__main__":
    import os
    os.chdir(r"D:\linyuejie\git-project\pet-reconstuction")
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
        device_id=0
    )
    ScatterCorrection(scanner_option, recon_option,
                 save_path=0, energy_resolution=0.15, tof_resolution=0,
                 attenuation_img=0, attenuation_img_size=0, activity_img=0, activity_img_size=0,
                 # params for simulation setting
                 desire_dimension=0, simulated_rings=0, simulated_detectors=0, sample_step=0, acceleration_factor=0, tof_range=0, energy_threshold=0, tof_bins=0)
