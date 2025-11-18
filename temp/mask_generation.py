import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

skull_seg = sitk.GetArrayFromImage(sitk.ReadImage("/Volumes/11PHB_data4/acquisition/20251027/splits/Skull_seg.nii"))
brain_body_seg = sitk.GetArrayFromImage(sitk.ReadImage("/Volumes/11PHB_data4/acquisition/20251027/splits/Brain_Body_seg.nii"))
brain_seg = sitk.GetArrayFromImage(sitk.ReadImage("/Volumes/11PHB_data4/acquisition/20251027/splits/Brain_seg.nii"))

density_mask = np.zeros_like(skull_seg)
density_mask[skull_seg == 1] = 1.61
density_mask[brain_seg == 1] = 1.04
density_mask[(brain_seg == 0) & (brain_body_seg == 1)] = 1
density_mask.tofile("/Volumes/11PHB_data4/acquisition/20251027/splits/20251027_human_density.raw")

index_mask = np.zeros_like(skull_seg)
index_mask[skull_seg == 1] = 10
index_mask[brain_seg == 1] = 11
index_mask[(brain_seg == 0) & (brain_body_seg == 1)] = 12
index_mask.tofile("/Volumes/11PHB_data4/acquisition/20251027/splits/20251027_human_index.raw")
