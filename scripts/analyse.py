import os
import os.path

import numpy as np
from skimage import io

voxel_unit = "µm³"
voxel_size = 7.536e-6


def to_human_vol(vol):
    return f"{round(vol * voxel_size, 3)}{voxel_unit}"


def analyse_segmentation(folder_path):
    file_name = os.path.basename(os.path.normpath(folder_path))

    cell = io.imread(f"{folder_path}/0_{file_name}.tif")
    vac = io.imread(f"{folder_path}/1_{file_name}.tif")
    lipid = io.imread(f"{folder_path}/2_{file_name}.tif")

    cell_sum = np.sum(cell)
    cell_vol = to_human_vol(cell_sum)
    vac_sum = np.sum(vac)
    vac_vol = to_human_vol(vac_sum)
    lipid_sum = np.sum(lipid)
    lipid_vol = to_human_vol(lipid_sum)

    overlap = vac * lipid  # by multiplying, only places where both values are 1 will remain.
    difference = lipid - overlap
    overlap_sum = np.sum(overlap)
    overlap_vol = to_human_vol(overlap_sum)
    dif_vol = to_human_vol(np.sum(difference))
    inside = np.around((overlap_sum / lipid_sum) * 100, 2)

    print("-" * 48)
    print(folder_path)
    print(f"Volumes: \t{cell_vol} \t{vac_vol} \t{lipid_vol}")
    print(f"In/out: \t{overlap_vol} \t{dif_vol} \t{inside}%")



if __name__ == "__main__":
    fol = "../yeast_segs"

    for cat in os.listdir(fol):
        for tomo_fol in os.listdir(f"{fol}/{cat}"):
            if os.path.isdir(f"{fol}/{cat}/{tomo_fol}"):
                analyse_segmentation(f"{fol}/{cat}/{tomo_fol}")
