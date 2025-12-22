import numpy as np

def trl_curvature(yi, bi, ri, li, ctype=None):
    """
    为泊松传输模型（Poisson transmission model）计算替代抛物线曲率（surrogate parabola curvatures）
    移植自 Matlab 同名函数，保持完全兼容

    Parameters:
        yi (np.ndarray): 观测值（如探测器光子计数）
        bi (np.ndarray or float): 背景项/系统参数（需为正）
        ri (np.ndarray or float): 散射项/残余计数（需非负）
        li (np.ndarray): 当前迭代的线衰减系数积分（line integral）
        ctype (str, optional): 曲率计算模式，默认 'oc'
            - 'oc': Erdogan 最优曲率（默认）
            - 'pc': 预计算曲率（兼容 trpl3）
            - 'nc': 牛顿曲率（当前二阶导数）

    Returns:
        np.ndarray: 非负曲率数组（与 yi 同形状）

    Raises:
        ValueError: 输入参数不完整或模式无效
    """

    # 输入参数检查
    if ctype is None:
        ctype = 'oc'
    ctype = ctype.lower()
    valid_ctypes = ['oc', 'pc', 'nc']
    if ctype not in valid_ctypes:
        raise ValueError(f"无效的 ctype: {ctype}，必须是 {valid_ctypes} 之一")
    if len([x for x in [yi, bi, ri, li] if x is None]) >= 1:
        raise ValueError("输入参数 yi, bi, ri, li 不能为空")

    # 确保输入为 NumPy 数组（支持标量输入自动广播）
    yi = np.asarray(yi, dtype=np.float64)
    bi = np.asarray(bi, dtype=np.float64) if not np.isscalar(bi) else bi
    ri = np.asarray(ri, dtype=np.float64) if not np.isscalar(ri) else ri
    li = np.asarray(li, dtype=np.float64)

    # 定义似然函数 h 和其对 l 的一阶导数 dh（对应原 Matlab inline 函数）
    def h(y, b, r, l):
        """泊松似然函数核心项：h = y*log(b*exp(-l) + r) - (b*exp(-l) + r)"""
        bel = b * np.exp(-l)
        yb = bel + r
        return y * np.log(yb) - yb

    def dh(y, b, r, l):
        """h 对 l 的一阶导数：dh = (1 - y/(b*exp(-l) + r)) * b*exp(-l)"""
        bel = b * np.exp(-l)
        yb = bel + r
        return (1 - y / yb) * bel

    # -------------------------- 模式 1: 'oc' 最优曲率 --------------------------
    if ctype == 'oc':
        # 初始化最大曲率（l=0 时的曲率）
        ni_max = np.zeros_like(yi)
        bi_scalar = np.isscalar(bi)
        ri_scalar = np.isscalar(ri)

        if bi_scalar:
            # 标量 bi（必须为正）
            if bi <= 0:
                raise ValueError("标量 bi 必须大于 0")
            ni_max = bi * (1 - yi * ri / (bi + ri) ** 2)
        else:
            # 数组 bi：仅处理 bi>0 的位置
            i0 = bi > 0
            ni_max[i0] = 0  # 初始化有效区域
            if ri_scalar:
                rii = 1 if ri == 0 else ri  # 标量 ri 处理
            else:
                rii = ri[i0]  # 数组 ri 取对应有效区域
            # 计算有效区域的 ni_max
            denominator = (bi[i0] + rii) ** 2
            ni_max[i0] = bi[i0] * (1 - yi[i0] * rii / denominator)

        # 强制曲率非负
        ni_max = np.maximum(ni_max, 0)
        ni = ni_max.copy()

        # 数值精度处理：li < 0.1 时用 ni_max，否则计算实际曲率（与原 Matlab 一致）
        il0 = li < 0.1

        # 计算 tmp = h(li) - h(0) - li*dh(li)
        h_li = h(yi, bi, ri, li)
        h_0 = h(yi, bi, ri, np.zeros_like(li))
        dh_li = dh(yi, bi, ri, li)
        tmp = h_li - h_0 - li * dh_li

        # 对 li >=0.1 的区域更新曲率
        i = ~il0
        if np.any(i):
            li_i = li[i]
            tmp_i = np.maximum(tmp[i], 0)  # 确保分子非负
            ni[i] = 2 / (li_i ** 2) * tmp_i

    # -------------------------- 模式 2: 'pc' 预计算曲率 --------------------------
    elif ctype == 'pc':
        ni = np.zeros_like(yi)
        # 有效射线条件：yi > ri 且 ri >=0 且 bi >0
        if np.isscalar(bi):
            bi_arr = np.full_like(yi, bi)
        else:
            bi_arr = bi
        if np.isscalar(ri):
            ri_arr = np.full_like(yi, ri)
        else:
            ri_arr = ri

        ii = (yi > ri_arr) & (ri_arr >= 0) & (bi_arr > 0)
        # 计算有效区域曲率：(yi - ri)^2 / yi
        ni[ii] = (yi[ii] - ri_arr[ii]) ** 2 / yi[ii]

    # -------------------------- 模式 3: 'nc' 牛顿曲率 --------------------------
    elif ctype == 'nc':
        # bel = bi * exp(-li)，yb = bel + ri
        bel = bi * np.exp(-li)
        if np.isscalar(ri):
            yb = bel + ri
        else:
            yb = bel + ri
        # 牛顿曲率公式：(1 - ri*yi/yb²) * bel
        ni = (1 - ri * yi / (yb ** 2)) * bel

    # 最终强制所有曲率非负（与原 Matlab 一致）
    ni = np.maximum(ni, 0)
    return ni


if __name__ == "__main__":
    pass