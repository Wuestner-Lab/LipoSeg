import os

import numpy as np
from skimage import measure, io

from scripts.util import save_imagej_composite


def split_segmentation(instance_seg, fname):
    cell_instances, n_labels = measure.label(instance_seg[:, :, :, 0], background=0, return_num=True, connectivity=2)
    print(fname, n_labels)
    for i in range(1, n_labels + 1):
        cell_mask = (cell_instances == i).astype(int)
        out = np.copy(instance_seg)
        for c in range(instance_seg.shape[3]):  # for each channel, except original img
            out[:, :, :, c] = instance_seg[:, :, :, c] * cell_mask
        out = np.rollaxis(out, 3, 1)
        save_imagej_composite(f"{fname}_{i}.tif", out, ["blue", "red", "green"])


if __name__ == "__main__":
    fol = "../yeast_segs"

    for cat in os.listdir(fol):
        for tomo_fol in os.listdir(f"{fol}/{cat}"):
            if os.path.isdir(f"{fol}/{cat}/{tomo_fol}"):
                seg = io.imread(f"{fol}/{cat}/{tomo_fol}/instance_composite_{tomo_fol}.tif")
                split_segmentation(seg, f"{fol}/{cat}/{tomo_fol}/{tomo_fol}_split")
