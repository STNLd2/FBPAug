import numpy as np
from dpipe.im import normalize, zoom
from dpipe.dataset.segmentation import SegmentationFromCSV
from fbp_aug.fbp import slice_to_sin


class Dataset(SegmentationFromCSV):
    def __init__(self, data_path, modalities, new_spacing, target='target', metadata_rpath='metadata.csv'):
        super().__init__(data_path=data_path,
                         modalities=modalities,
                         target=target,
                         metadata_rpath=metadata_rpath)
        self.new_spacing = new_spacing

    def _raw_spacing(self, identifier):
        return self.df.loc[identifier, 'spacing']

    def _scale_factor(self, identifier):
        new_spacing = np.array(self.new_spacing, float)
        spacing = self._raw_spacing(identifier)
        new_spacing = np.where(~np.isnan(new_spacing), new_spacing, spacing)
        return spacing / new_spacing

    def spacing(self, identifier):
        return self.spacing(identifier) / self.scale_factor(identifier)

    def image(self, identifier):
        image = np.float32(super().load_image(identifier))
        return zoom(image, scale_factor=self.scale_factor)

    def lungs(self, identifier):
        lungs = np.float32(super().load_segm(identifier)[None])
        return zoom(lungs.astype(float), scale_factor=self.scale_factor)

    def sinogram(self, identifier):
        image = self.image(identifier)
        shape = image.shape
        sinograms = []
        for i in range(shape[-1]):
            slc = image[..., i]
            sinograms.append(slice_to_sin(slc, bins=shape[0]))
        return np.stack(sinograms, axis=-1).astype(np.float16)


def normalize_ct(image, dtype='float32', min_clip=-1350, max_clip=150, axis=None):
    image = np.clip(image, min_clip, max_clip)
    return normalize(image, dtype=dtype, axis=axis)
