import numpy as np
import skimage.morphology as morph
from scipy import ndimage as ndi
from scipy.spatial.distance import cityblock
from skimage import io
from skimage.segmentation import watershed


def connectivity_matrix(dims: tuple, full_connectivity: bool = False) -> np.ndarray:
    shape = tuple(3 for i in range(len(dims)))  # (3), (3,3), (3,3,3) etc
    if full_connectivity:
        return np.ones(shape)
    else:
        mat = np.zeros(shape)
        center = tuple(1 for i in range(len(dims)))
        for coord in np.ndindex(shape):
            if cityblock(coord, center) <= 1:
                mat[coord] = 1
        return mat


def binary_distance_transform(image: np.ndarray) -> np.ndarray:
    binary_img = (image > 0)
    dist_transf = np.array(ndi.distance_transform_edt(binary_img)).astype(int)
    return dist_transf


def invert_image(image: np.ndarray) -> np.ndarray:
    img = image.astype(int)
    img_max = np.max(img)
    return img_max - img


def labeled_components(minima: np.ndarray, footprint: np.ndarray) -> np.ndarray:
    min_val = max(np.min(minima), 1)  # minimum but at least 1, we don't want background.
    max_val = np.max(minima)
    labels_n = 0
    labeled_img = np.zeros(minima.shape)

    for i in range(min_val, max_val + 1):
        sub_img = (minima == i)
        label, n_feats = ndi.label(sub_img, footprint)
        label[label > 0] = label[label > 0] + labels_n  # add already found labels to newly found labels
        labeled_img += label
        labels_n += n_feats
    return labeled_img


def extendedmin(img, h):
    mask = img.copy()
    marker = mask + h
    hmin = morph.reconstruction(marker, mask, method='erosion')
    return morph.local_minima(hmin)


def imposemin(img, minima):
    marker = np.full(img.shape, np.inf)
    marker[minima == 1] = 0
    mask = np.minimum((img + 1), marker)
    return morph.reconstruction(marker, mask, method='erosion')


def extended_minima_watershed(image: np.ndarray, mask: np.ndarray, dynamic: int = 2,
                              full_connectivity: bool = False) -> np.ndarray:
    footprint = connectivity_matrix(image.shape, full_connectivity)
    minima = extendedmin(image, dynamic)
    imposed = imposemin(image, minima)
    labels = labeled_components(minima, footprint)

    basins = watershed(imposed, labels, footprint, mask=mask, watershed_line=True)
    return basins


def distance_transform_watershed(image: np.ndarray, dynamic: int = 2, full_connectivity: bool = False) -> np.ndarray:
    dist_map = binary_distance_transform(image)
    inverted = invert_image(dist_map)
    ws = extended_minima_watershed(inverted, image, dynamic, full_connectivity)
    return ws


def main():
    image = io.imread(
        "C:\\Users\\LAB-ADMIN\\Desktop\\pipeline-program\\yeast_segs\\wt\\tomo_20210317-03\\0_tomo_20210317-03.tif")
    ws = distance_transform_watershed(image, dynamic=8)
    io.imsave("out2.tif", ws)


if __name__ == "__main__":
    main()
