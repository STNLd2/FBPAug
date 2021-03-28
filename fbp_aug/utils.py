import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from dpipe.io import load_json, save_numpy


def get_results(path, metric='dice', metrics_dir='test_metrics'):
    from os.path import exists, join as jp
    scores = []
    ids = []
    i = 0
    while exists(jp(path, f'experiment_{i}')):
        temp_json = load_json(f'{path}/experiment_{i}/{metrics_dir}/{metric}.json')
        scores.extend(list(temp_json.values()))
        ids.extend(list(temp_json.keys()))
        i += 1
    return np.array(scores), np.array(ids), i


def get_contours(mask):
    from skimage.measure import find_contours

    n = mask.shape[-1]
    contours = dict()

    for i in range(n):
        try:
            x = np.concatenate(find_contours(mask[..., i], .1))
            contours[i] = x
        except:
            pass
    return contours


def slice_contours(img, mask1, mask2):
    from ipywidgets import interact, IntSlider
    contour1 = get_contours(mask1)
    contour2 = get_contours(mask2)

    def update(idx):
        fig, axes = plt.subplots(1, 1, figsize=(7, 7))

        axes.imshow(img.take(idx, axis=-1), cmap='gray')

        try:
            c1 = contour1[idx].T

            axes.plot(c1[1], c1[0], 'ro', ms=1)
        except:
            pass
        try:
            c2 = contour2[idx].T
            axes.plot(c2[1], c2[0], 'ro', ms=1, color='c')
        except:
            pass

        axes.set_axis_off()

        plt.tight_layout()
        plt.show()

    interact(update, idx=IntSlider(min=0, max=img[0].shape[-1] - 1, continuous_update=False))


def np_sigmoid(x):
    """Applies sigmoid function to the incoming value(-s)."""
    return 1 / (1 + np.exp(-x))


def volume2diameter(volume):
    return (6 * volume / np.pi) ** (1 / 3)


def diameter2volume(diameter):
    return np.pi / 6 * diameter ** 3


def fix_seed(seed=0xBadCafe):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_binary_mask(value, path, *, allow_pickle: bool = True, fix_imports: bool = True,
                     compression: int = None):
    save_numpy(value >= 0.5, path, allow_pickle=allow_pickle, fix_imports=fix_imports, compression=compression)
