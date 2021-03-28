from functools import partial

import numpy as np
from scipy.fftpack import fft, ifft
from skimage.transform import radon
from skimage.transform.radon_transform import _sinogram_circle_to_square


def hu_to_mm(x):
    water = 1e-4
    air = 1.2041e-6
    return x / 1000 * (water - air) + water


def mm_to_hu(x):
    water = 1e-4
    air = 1.2041e-6
    return (x - water) * 1000 / (water - air)


def slice_to_sin(slc, bins=512):
    slc = slc.copy()
    slc[slc <= -1000] = -1000
    slc = hu_to_mm(slc)
    theta = np.linspace(0, 180, bins, endpoint=False)
    return radon(slc, theta)


def sin_to_slice(sin, bins=512, a=0, b=1):
    theta = np.linspace(0, 180, bins, endpoint=False)
    slc = iradon(sin, theta=theta, a=a, b=b)
    slc = mm_to_hu(slc)
    slc[slc <= -1000] = -1000
    slc[slc >= 3000] = 3000
    return slc


def ramp_filter(size):
    n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=np.int),
                        np.arange(size / 2 - 1, 0, -2, dtype=np.int)))
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    fourier_filter = 2 * np.real(fft(f))
    return fourier_filter.reshape(-1, 1)


def smooth_sharpen_filter(size, a, b):
    ramp = ramp_filter(size)
    return ramp * (1 + a * (ramp ** b))


def iradon(radon_image, a=0, b=1, theta=None):
    """
    adopted from scikit-image
    :param radon_image: sinogram
    :param a, b: transformation parameters
    :return: 2D image
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
    fourier_filter = smooth_sharpen_filter(projection_size_padded, a, b)
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


def apply_conv_filter(slc, a=0, b=1, bins=None):
    assert len(slc.shape) == 2
    assert slc.shape[0] == slc.shape[1]
    if bins is None:
        bins = slc.shape[0]
    sin = slice_to_sin(slc, bins=bins)
    return sin_to_slice(sin, a=a, b=b, bins=bins)
