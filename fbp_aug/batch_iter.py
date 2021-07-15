import numpy as np

from skimage.util import random_noise
from skimage.exposure import adjust_gamma

from fbp_aug.dataset import scale_ct
from fbp_aug.fbp import apply_conv_filter

SPATIAL_DIMS = (-3, -2, -1)


def extract_slice(inputs, non_zero_fraction=0.5):
    *spatial_inputs, z_indices = inputs
    z_shape = spatial_inputs[0].shape[-1]
    if np.random.rand() <= non_zero_fraction:
        z_center = np.random.choice(z_indices)
    else:
        z_center = np.random.choice(z_shape)
    outputs = [inp[..., z_center] for inp in spatial_inputs]
    return outputs


# ################### augmentation ####################################################################################


def symmetry_transform_2d(inputs, spatial2d_axis=(-2, -1)):
    k = np.random.randint(0, 3)
    flip = np.random.rand() >= 0.5

    outputs = tuple(np.rot90(inp, k, axes=spatial2d_axis) for inp in inputs)
    if flip:
        outputs = tuple(np.flip(out, axis=-2) for out in outputs)

    return outputs


def gaussian_noise_with_scaling(img, p=0.5):
    img = scale_ct(img)
    if np.random.rand() <= p:
        img = random_noise(img, mode='gaussian', clip=False)
    return np.clip(img, 0, 1)


def gamma_with_scaling(img, gamma=0.1, p=0.5):
    img = scale_ct(img)
    if np.random.rand() <= p:
        gamma = np.exp(np.random.randn() * gamma)
        return adjust_gamma(img, gamma=gamma)
    return img


def windowing_with_scaling(image, center=-600, width=1500, p=0.5):
    if np.random.rand() <= p:
        center = center + np.random.randint(-100, 100)
        window_size = width + np.random.randint(-200, 200)
        min_clip = int(center - window_size / 2)
        max_clip = int(center + window_size / 2)
        return scale_ct(image, min_clip=min_clip, max_clip=max_clip)
    return scale_ct(image)


def fbp_with_scaling(img, p=0.5):
    if np.random.rand() >= p:
        return scale_ct(img)
    shape = img.shape
    if np.random.rand() < 0.5:   # smoothing
        a = (-1.0) * np.random.rand()
        b = 0.1 + 0.9 * np.random.rand()
    else:                        # sharpening
        a = 10 + 30 * np.random.rand()
        b = 1.0 + 3 * np.random.rand()

    apply_conv_filter(img.reshape(shape[-2], shape[-1]), a=a, b=b, bins=shape[-1] // 2)

    return scale_ct(img.reshape(shape))
