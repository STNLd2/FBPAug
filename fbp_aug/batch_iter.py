import random

import numpy as np

from dpipe.im import crop_to_box
from dpipe.im.box import get_centered_box
from skimage.util import random_noise
from skimage.transform import rotate
from skimage.exposure import adjust_gamma

from dpipe.im.patch import get_random_patch, sample_box_center_uniformly

from fbp_aug.dataset import normalize_ct
from fbp_aug.fbp import sin_to_slice, apply_conv_filter

SPATIAL_DIMS = (-3, -2, -1)


def extract_slice(inputs, non_zero_fraction=0.5):
    if np.random.rand() >= non_zero_fraction:
        slices = get_random_patch(*inputs, patch_size=1)
        return tuple(np.moveaxis(slc, -1, 0) for slc in slices)

    slices = np.unique(np.where(inputs[-1])[-1])
    z = np.random.choice(slices)
    return tuple(array[..., z][None] for array in inputs)


def center_choice(inputs, center_margin=5, nonzero_fraction=0.5, shift_value=(17, 17, 5)):
    """`centers` comes last."""
    *spatial_inputs, centers = inputs
    y_shape = np.array(spatial_inputs[1].shape)[np.array(SPATIAL_DIMS)]

    if len(centers) > 0 and np.random.uniform() < nonzero_fraction:
        center = random.choice(centers)
        if isinstance(shift_value, int):
            shift_value = [shift_value] * 3
        center += np.array([np.random.randint(-v, v) if v > 0 else 0 for v in shift_value])
        center = np.array([np.clip(c, 0, s - 1) for c, s in zip(center, y_shape)])
    else:
        center = np.int64([np.random.randint(center_margin, s - 1 - center_margin) for s in y_shape])

    return (*spatial_inputs, center)


def fbp3d(inputs, p=0.5):
    img, target, sin = inputs

    if np.random.rand() >= p:
        return img, target
    else:

        if np.random.rand() < 0.5:   # smoothing
            a = (-0.5) * np.random.rand()
            b = 0.1 + 0.9 * np.random.rand()
        else:                        # sharpening
            b = 1.0 + 3 * np.random.rand()
            a = b * 10 * (np.random.rand() + 0.5)

        # shape2d = img[..., 0].shape
        img_new = np.zeros_like(img)
        for z in range(img.shape[-1]):
            img_new[..., z] = sin_to_slice(sin[..., z], a=a, b=b, bins=sin[..., z].shape[-1])

        return img_new, target


def sample_center_uniformly(shape, patch_size, spatial_dims=SPATIAL_DIMS):
    spatial_shape = np.array(shape)[list(spatial_dims)]
    if np.all(patch_size <= spatial_shape):
        return sample_box_center_uniformly(shape=spatial_shape, box_size=patch_size)
    else:
        return spatial_shape // 2


def extract_final_patch(inputs, patch_sizes, padding_values, axis=SPATIAL_DIMS):
    """`centers` comes last."""
    center = sample_center_uniformly(inputs[1].shape, patch_size=patch_sizes[1], spatial_dims=axis)
    spatial_outputs = (crop_to_box(inp, box=get_centered_box(center, np.array(patch)), padding_values=pad, axis=axis)
                       for inp, patch, pad in zip(inputs, patch_sizes, padding_values))
    return spatial_outputs


# ################### augmentation ####################################################################################


def symmetry_transform_2d(inputs, spatial2d_axis=(-2, -1)):
    k = np.random.randint(0, 3)
    flip = np.random.rand() >= 0.5

    outputs = tuple(np.rot90(inp, k, axes=spatial2d_axis) for inp in inputs)
    if flip:
        outputs = tuple(np.flip(out, axis=-2) for out in outputs)

    return outputs


def add_gaussian_noise(img):
    img, target = img
    return random_noise(img, mode='gaussian', clip=False), target


def gamma_transform(img, already_scaled=False):
    img, target = img
    gamma = np.exp(np.random.randn() * 0.2)
    if not already_scaled:
        img_max, img_min = np.max(img), np.min(img)
        scaled_image = (img - img_min) / (img_max - img_min)
        gamma_scaled_img = adjust_gamma(scaled_image, gamma=gamma)
        return gamma_scaled_img * (img_max - img_min) + img_min, target
    else:
        return adjust_gamma(img, gamma=gamma), target


def windowing_augmentation(image, apply_normalize=True):
    image, target = image
    center = -600 + np.random.randint(-100, 100)
    window_size = 1500 + np.random.randint(-200, 200)
    min_clip = int(center - window_size / 2)
    max_clip = int(center + window_size / 2)
    if apply_normalize:
        return normalize_ct(image, min_clip=min_clip, max_clip=max_clip), target
    else:
        return np.clip(image, a_min=min_clip, a_max=max_clip), target


def fbp(inputs, p=0.5):
    img, target, sin = inputs
    if np.random.rand() >= p:
        return img, target
    shape = img.shape
    if np.random.rand() < 0.5:   # smoothing
        a = (-1.0) * np.random.rand()
        b = 0.1 + 0.9 * np.random.rand()
    else:                        # sharpening
        a = 10 + 30 * np.random.rand()
        b = 1.0 + 3 * np.random.rand()

    apply_conv_filter(img.reshape(shape[-2], shape[-1]), a=a, b=b, bins=shape[-1] // 2)

    return img.reshape(shape), target
