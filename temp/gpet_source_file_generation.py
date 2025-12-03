import numpy as np
import matplotlib.pyplot as plt
import os

def zero_outside_radius(image, radius, center=None):
    """
    将二维图像中指定半径外的元素置零

    参数：
        image: 二维numpy数组（输入图像）
        radius: 保留区域的半径（距离中心小于等于该值的区域保留）
        center: 中心坐标(tuple)，格式为(cx, cy)，默认使用图像中心点

    返回：
        处理后的二维数组（半径外元素为0）
    """
    # 获取图像尺寸（行数、列数）
    rows, cols = image.shape

    # 确定中心坐标（默认中心点，支持浮点数坐标提高精度）
    if center is None:
        cx, cy = rows / 2.0, cols / 2.0  # 中心点坐标（浮点数）
    else:
        cx, cy = center  # 自定义中心（可输入整数或浮点数）

    # 生成所有像素的坐标网格（行索引i，列索引j）
    # indexing='ij'确保ii是行方向网格，jj是列方向网格
    i = np.arange(rows)
    j = np.arange(cols)
    ii, jj = np.meshgrid(i, j, indexing='ij')  # 形状均为(rows, cols)

    # 计算每个像素到中心的距离平方（避免开方，提升效率）
    dist_sq = (ii - cx) ** 2 + (jj - cy) ** 2  # 形状为(rows, cols)

    # 创建掩码：距离平方 > 半径平方 → 需要置零的区域
    mask = dist_sq > radius ** 2

    # 复制原图像（避免修改输入数组），并将掩码区域置零
    result = image.copy()
    result[mask] = 0

    return result


def voxel_center_distance(shape, voxel_size):
    """
    计算三维数组中每个体素中心相对于数组中心的物理距离

    参数：
        shape: 三维数组形状，tuple类型，格式为(D, H, W)
        voxel_size: 体素大小，tuple类型，格式为(vx, vy, vz)（对应3个维度的物理尺寸）

    返回：
        distance: 与输入数组同形状的三维数组，每个元素为对应体素中心到数组中心的物理距离
    """
    # 1. 获取数组维度和体素大小
    D, H, W = shape
    vz, vy, vx = voxel_size

    # 2. 计算数组中心的物理坐标（体素中心对齐）
    # 索引空间中心：各维度大小的一半减0.5（体素中心在索引[i, j, k]的中心位置）
    center_idx = (D / 2, H / 2, W / 2)
    # 转换为物理坐标：索引×体素大小
    center_phys = (center_idx[0] * vz, center_idx[1] * vy, center_idx[2] * vx)

    # 3. 生成所有体素中心的索引坐标网格
    d = np.arange(D)  # 深度维度索引（0到D-1）
    h = np.arange(H)  # 高度维度索引（0到H-1）
    w = np.arange(W)  # 宽度维度索引（0到W-1）
    # 生成三维网格，indexing='ij'确保网格形状与数组一致
    dd, hh, ww = np.meshgrid(d, h, w, indexing='ij')

    # 4. 转换为物理坐标（每个体素中心的物理位置）
    phys_d = dd * vz
    phys_h = hh * vy
    phys_w = ww * vx

    # 5. 计算每个体素到数组中心的物理偏移量
    dz = phys_d - center_phys[0]
    dy = phys_h - center_phys[1]
    dx = phys_w - center_phys[2]

    # 6. 计算欧氏距离（sqrt(dx² + dy² + dz²)）
    # distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    return dx, dy, dz


def print_source_file(index, source_particle, location, dir_path):
    file_path = dir_path + "sources_%d.txt" % index
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(f"{np.sum(source_particle > 0)}\n")
        f.write(f"Natoms(equal to activity times lamda) type(from isotope file, start from 0), shape type, centerx(cm) centery(cm) centerz(cm) Threecoefftodefineshape# \n")
        for i in range(source_particle.shape[0]):
            f.write(f"{(source_particle[i]).astype(int)} {0} {0} {location[i, 0]} {location[i, 1]} {location[i, 2]} {0.13} {0.13} {0.279}\n")


def print_input_file(index, dir_path):
    file_path = dir_path + "input_%d.in" % index
    # 定义需要替换的变量（可根据实际值修改）
    device_number = 3
    noncolinear_angle = 0.0037056
    phantom_dimension = (384, 384, 71)
    phantom_offset = (-24.96, -24.96, -48.82)
    phantom_size = (49.92, 49.92, 19.809)
    phantom_material_file = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/input/phantom_20251027_human/20251027_human_index.raw"
    phantom_density_file = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/input/phantom_20251027_human/20251027_human_density.raw"
    simulation_history = 0
    use_phase_space_source = 0
    source_file = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/input/source_20251027_human_fix/sources_%d.txt" % index
    particle_type = 0
    positron_range_consider = 0
    decay_time = (0, 100)
    photon_psf_sphere = (0, 0, 0, 5)
    photon_absorption_energy = 1e3
    detector_geo_file = "/share/home/lyj/Downloads/gPET_to_SZBL/Example/input/scanner/PET_11Panel_LD_crystal_base.geo"
    quadratic_surfaces_num = 1
    quadratic_surfaces_params = (0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
    readout_depth_policy = (2, 1)
    thresholder_energy_deadtime = 50000
    energy_blur_params = (1, 511000, 0.05, 0, 0)
    deadtime_params = (3, 0, 2.2)
    energy_window = (410000, 610000)

    # 构造格式化字符串（保持原文本结构）
    formatted_text = f"""device number:
{device_number}
noncolinear angle (sigma of Gaussian in rad):
{noncolinear_angle}
phantom dimension
{phantom_dimension[0]} {phantom_dimension[1]} {phantom_dimension[2]}
phantom offset in global coordinate (in cm):
{phantom_offset[0]} {phantom_offset[1]} {phantom_offset[2]}
phantom size (in cm):
{phantom_size[0]} {phantom_size[1]} {phantom_size[2]}
phantom material data file:
{phantom_material_file}
phantom density data file:
{phantom_density_file}
simulation history: (only use it when set `use phase space file as source = 1`)
{simulation_history}
use phase space file as source (0 for no, 1 for yes):
{use_phase_space_source}
source file:
{source_file}
particle type of phase-space file for source (0 for positron, 1 for photon):
{particle_type}
positron range consideration (0 for no, 1 for yes):
{positron_range_consider}
start time, end time for the radioactive decay process (in seconds):
{decay_time[0]} {decay_time[1]}
center (x, y, z) and radius for the photon-PSF-recording spherical surface in the global coordinate (in cm):
{photon_psf_sphere[0]} {photon_psf_sphere[1]} {photon_psf_sphere[2]} {photon_psf_sphere[3]}
photon absorption energy (in eV):
{photon_absorption_energy}
detector geometry file:
{detector_geo_file}
total number of quadratic surfaces and the TEN parameters (x2 y2 z2 xy xz yz x y z constant) to define each quadratic surface in the local coordinate (in cm):
{quadratic_surfaces_num}
{' '.join(map(str, quadratic_surfaces_params))}
readout depth (1 to 3), policy (1 and 2):
{readout_depth_policy[0]} {readout_depth_policy[1]}
thresholder energy before deadtime (in eV):
{thresholder_energy_deadtime}
policy, reference energy (in eV), reference resolution (in eV) and slope for energy blurring, and resolution for space blurring (in cm):
{' '.join(map(str, energy_blur_params))}
deadtime level (1 to 3), deadtime policy (0 for paralyzable, 1 for nonparalyzable), and deadtime duration (in micro-second):
{' '.join(map(str, deadtime_params))}
thresholder and upholder for energy window (in eV):
{energy_window[0]} {energy_window[1]}
    """

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)


if __name__ == "__main__":
    img = np.fromfile("/Users/seanlam/Downloads/recon_20251027_t4_human_wrr_wtof_wengwin_410_610_wnorm_watn_ge_size_it6.img", dtype=np.float32).reshape([71, 384, 384])
    for i in range(71):
        img[i, :, :] = zero_outside_radius(img[i, :, :], radius=90)

    img /= img.max()
    img[img < 0.05] = 0
    slice_sum = img.sum(axis=1).sum(axis=1)
    max_slice = slice_sum.argmax()
    ratio = np.floor(2**31 / slice_sum.max())  # currently is 435429

    particle_num = img.copy() * ratio
    particle_sum = particle_num.sum(axis=1).sum(axis=1).max()
    dx, dy, dz = voxel_center_distance(img.shape, [0.279, 0.13, 0.13])

    '''
    目前按照每个文件包含一个 slice 的方法只能得到 500w 左右的数据，需要 500w x 2 = 1000w 左右的数据量
    需要根据仿真光子数量自动分片
    '''
    particle_num *= 2 / 10 * 4
    particle_num = particle_num.reshape(-1).astype(int)
    print(particle_num.sum())
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)
    dz = dz.reshape(-1)

    dx = dx[particle_num > 0]
    dy = dy[particle_num > 0]
    dz = dz[particle_num > 0]
    particle_num = particle_num[particle_num > 0]

    splits_index = np.zeros_like(particle_num)
    particle_num_copy = particle_num.copy()
    left_index = 0
    for i in range(999):
        cum_sum = np.cumsum(particle_num_copy)
        tail_judge = np.where(cum_sum > 2**30)[0]
        if tail_judge.shape[0] != 0:
            right_index = tail_judge[0]
            splits_index[left_index:right_index] = i
            particle_num_copy[left_index:right_index] = 0
            left_index = right_index
        else:
            right_index = particle_num_copy.shape[0]
            splits_index[left_index:right_index] = i
            break

    for i in range(splits_index.max()):
        current_slice = particle_num[splits_index == i]
        current_dx = dx[splits_index == i]
        current_dy = dy[splits_index == i]
        current_dz = dz[splits_index == i]

        source_dir = "/Users/seanlam/Documents/WorkInSZBL/gpet_from_pku/gPET_to_SZBL/Example/input/source_20251027_human_fix/"
        input_dir = "/Users/seanlam/Documents/WorkInSZBL/gpet_from_pku/gPET_to_SZBL/Example/input/input_20251027_human_crystal_base_fix/"
        if not os.path.exists(source_dir):
            os.makedirs(source_dir)
        if not os.path.exists(input_dir):
            os.makedirs(input_dir)
        print_source_file(i, current_slice, np.round(np.column_stack((current_dx, current_dy, current_dz)), 4), source_dir)
        print_input_file(i, input_dir)

