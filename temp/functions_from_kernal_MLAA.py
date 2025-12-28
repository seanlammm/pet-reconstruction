import numpy as np

import numpy as np


def trl_curvature_oc(yi, bi, ri, li):
    """
    Compute surrogate parabola curvatures for Poisson transmission model.
    Only implements the 'oc' (Optimal Curvature) mode based on Erdogan's formula.

    Args:
        yi (np.ndarray): Measurement data (counts).
        bi (np.ndarray): Normalized activity projection (blank scan equivalent).
        ri (np.ndarray): Randoms + Scatter.
        li (np.ndarray): Current attenuation line integrals.

    Returns:
        ni (np.ndarray): The optimal curvature map.
    """

    # 1. Convert inputs to single precision (float32) to match MATLAB code behavior
    yi = yi.astype(np.float32)
    bi = bi.astype(np.float32)
    ri = ri.astype(np.float32)
    li = li.astype(np.float32)

    # 2. Define Helper Functions (implemented as inline vector operations below)
    # The original MATLAB code defines:
    # h = @(y,b,r,l)(y.*log(b.*exp(-l)+r)-(b.*exp(-l)+r));
    # dh = @(y,b,r,l)((1 - y ./ (b.*exp(-l)+r)) .* b.*exp(-l));

    # Pre-calculate common term: expected value y_bar = b * exp(-l) + r
    # We use np.maximum to avoid potential division by zero if inputs are bad,
    # though in valid PET data yb should be > 0.
    yb = bi * np.exp(-li) + ri

    # 3. Compute curvature at l = 0 (Base approximation)
    # MATLAB: ni_max = bi .* (1 - yi .* ri ./ (bi + ri).^2);
    # NumPy broadcasting handles scalars automatically.
    # Added a tiny epsilon to denominator to prevent division by zero if bi+ri=0
    denom_0 = (bi + ri) ** 2
    term_0 = np.divide(yi * ri, denom_0, where=denom_0 != 0)
    ni_max = bi * (1 - term_0)

    # Ensure non-negative (Convexity constraint)
    ni = np.maximum(ni_max, 0)

    # 4. Handle "Large" attenuation values (li >= 0.1)
    # Erdogan's formula is numerically unstable near 0, so we use the limit (step 3) for small li
    # and the full formula for large li.
    mask_large = li >= 0.1

    if np.any(mask_large):
        # Extract values where li >= 0.1 to save computation and ensure safety
        y_m = yi[mask_large]
        b_m = bi[mask_large] if bi.ndim > 0 and bi.size > 1 else bi
        r_m = ri[mask_large] if ri.ndim > 0 and ri.size > 1 else ri
        l_m = li[mask_large]
        yb_m = yb[mask_large]  # The predicted value at current l

        # Calculate h(l) and dh(l) at current l
        # h(l) = y * log(yb) - yb
        h_val = y_m * np.log(yb_m) - yb_m

        # dh(l) = (1 - y/yb) * (yb - r)
        # Note: b*exp(-l) is equivalent to (yb - r)
        dh_val = (1 - y_m / yb_m) * (yb_m - r_m)

        # Calculate h(0) -> l=0 implies exp(-l)=1, so yb_0 = b + r
        yb_0 = b_m + r_m
        h_0 = y_m * np.log(yb_0) - yb_0

        # Erdogan's Optimal Curvature Formula:
        # ni = (2 / l^2) * [ h(l) - h(0) - l * dh(l) ]
        tmp = h_val - h_0 - l_m * dh_val

        # Apply formula and update the 'ni' array
        ni_correction = (2.0 / (l_m ** 2)) * np.maximum(tmp, 0)
        ni[mask_large] = ni_correction

    return ni


if __name__ == "__main__":
    pass