import numpy as np
from dpipe.im import normalize


def scale_ct(image, dtype='float32', min_clip=-1350, max_clip=150, axis=None):
    image = np.clip(image, min_clip, max_clip)
    return normalize(image, dtype=dtype, axis=axis)
