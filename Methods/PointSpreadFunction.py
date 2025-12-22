import numpy as np
from scipy.ndimage import gaussian_filter


class PointSpreadFunction:
    def __init__(self, sigma):
        self.sigma = sigma

    def add_static_psf(self, img):
        gaussian_filter(img, self.sigma, order=0, output=img, mode='reflect', radius=None)
        return img
