from functools import partial
from typing import Union

import numpy as np
from scipy.fftpack import fft, ifft
from skimage.transform import radon
from skimage.transform.radon_transform import _sinogram_circle_to_square


def apply_conv_filter(slc: np.ndarray, a: float = 0, b: float = 1,
                      bins: Union[np.ndarray, int] = None) -> np.ndarray:
    """
    Performs transform, proposed in paper with given parameters a and b.
    Appropriate parameters for smoothing: a in (-1, 0), b in (0, 1)
                           for sharpening: a in (10, 40), b in (1, 4)
    slc
        whole slice of a CT image, should be 2D with values are in Hounsfield units and should be square
    a, b
        transformation parameters
    bins
        if int, number of projections' angles uniformly from 0 to 180 degrees,
        if np.ndarray, projection angles
        None is same as ``bins = slc.shape[0]``
    """
    assert slc.ndim == 2
    assert slc.shape[0] == slc.shape[1]
    if bins is None:
        bins = slc.shape[0]
    sin = _slice_to_sin(slc, bins=bins)
    return _sin_to_slice(sin, a=a, b=b, bins=bins)


def iradon(radon_image: np.ndarray, a: float = 0, b: float = 1,
           theta: Union[int, np.ndarray] = None) -> np.ndarray:
    """
    Adopted from scikit-image. Apply FBP with custom kernel to a sinogram ``radon_image``
    radon_image
        sinogram
    a, b
        transformation parameters
    """
    if radon_image.ndim != 2:
        raise ValueError('The input image must be 2-D')

    if theta is None:
        theta = np.linspace(0, 180, radon_image.shape[1], endpoint=False)

    angles_count = len(theta)
    if angles_count != radon_image.shape[1]:
        raise ValueError("The given ``theta`` does not match the number of "
                         "projections in ``radon_image``.")

    img_shape = radon_image.shape[0]
    output_size = img_shape

    radon_image = _sinogram_circle_to_square(radon_image)
    img_shape = radon_image.shape[0]

    # Resize image to next power of two (but no less than 64) for
    # Fourier analysis; speeds up Fourier and lessens artifacts
    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_shape))))
    pad_width = ((0, projection_size_padded - img_shape), (0, 0))
    img = np.pad(radon_image, pad_width, mode='constant', constant_values=0)
    # Apply filter in Fourier domain
    fourier_filter = _smooth_sharpen_filter(projection_size_padded, a, b)
    projection = fft(img, axis=0) * fourier_filter
    radon_filtered = np.real(ifft(projection, axis=0)[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros((output_size, output_size))
    radius = output_size // 2
    xpr, ypr = np.mgrid[:output_size, :output_size] - radius
    x = np.arange(img_shape) - img_shape // 2

    for col, angle in zip(radon_filtered.T, np.deg2rad(theta)):
        t = ypr * np.cos(angle) - xpr * np.sin(angle)
        interpolant = partial(np.interp, xp=x, fp=col, left=0, right=0)
        reconstructed += interpolant(t)

    out_reconstruction_circle = (xpr ** 2 + ypr ** 2) > radius ** 2
    reconstructed[out_reconstruction_circle] = 0.

    return reconstructed * np.pi / (2 * angles_count)


def _hu_to_mm(x):
    water = 1e-4
    air = 1.2041e-6
    return x / 1000 * (water - air) + water


def _mm_to_hu(x):
    water = 1e-4
    air = 1.2041e-6
    return (x - water) * 1000 / (water - air)


def _slice_to_sin(slc, bins=512):
    slc = slc.copy()
    slc[slc <= -1000] = -1000
    slc = _hu_to_mm(slc)
    theta = np.linspace(0, 180, bins, endpoint=False)
    return radon(slc, theta)


def _sin_to_slice(sin, bins=512, a=0, b=1):
    theta = np.linspace(0, 180, bins, endpoint=False)
    slc = iradon(sin, theta=theta, a=a, b=b)
    slc = _mm_to_hu(slc)
    slc[slc <= -1000] = -1000
    slc[slc >= 3000] = 3000
    return slc


def _ramp_filter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                        np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fft(f))
    return fourier_filter.reshape(-1, 1)


def _smooth_sharpen_filter(size, a, b):
    ramp = _ramp_filter(size)
    return ramp * (1 + a * (ramp ** b))
