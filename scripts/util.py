import argparse
import math
import struct

import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage import measure
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tifffile import tifffile


def progress_bar(current, total, bar_length=20, verbose=True):
    """
     Prints a progress bar in STDOUT with progress based on para
     ----------
     current : int
         current 'work' done
     total : int
        total 'work' to do
     verbose : bool
        Whether to output or not
     """
    if verbose:
        fraction = current / total

        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '

        ending = '\n' if current == total else '\r'

        print(f'Progress: [{arrow}{padding}] {current}/{total}', end=ending)


def log_msg(msg, verbose=True):
    """
     Prints a message, if verbose is not disabled
     ----------
     verbose : bool
        Whether to output or not
     """
    if verbose:
        print(msg)


# generate normal distribution with 0 mean and 0.5 std dev for a 512x512 image
quantiles = np.sort(np.random.normal(0, 0.5, 512 * 512))


def quantile_normalize(img, distribution=quantiles):
    """
    quantile normalizes a 512x512 image / matrix using the given distribution
    ----------
    img : np.array
        512x512 matrix to normalize
    distribution : np.array
        a sorted list with 512*512 values to use for normalization
    Returns
    -------
    np.array
        normalized image
     """
    order = np.argsort(img.flatten())  # get the rankings of the flattened frame
    return np.take_along_axis(distribution, order, 0).reshape((512, 512))  # take quantiles values based on ordered rankings, then reshape the image.


# returns img normalized to random normal distribution with 0 mean and 0.5 std dev for a 512x512 image
def random_quantile_normalize(img):
    """
    quantile normalizes a 512x512 image / matrix using a randomly generated
    normal distribution with 0 mean and 0.5 standard deviation
    ----------
    img : np.array
        512x512 matrix to normalize
    Returns
    -------
    np.array
        normalized image
     """
    return quantile_normalize(img, distribution=np.sort(np.random.normal(0, 0.5, 512 * 512)))


def normalize_image(img):
    """
    Squeezes values in a(n) (image) matrix between 0 and 1
    ----------
    img : np.array
        image to normalize
    Returns
    -------
    np.array
        normalized image
     """
    # lower bound to 0
    i_min = np.min(img)
    if i_min != 0:
        img -= i_min

    # clamp values between 0 and 1
    i_max = np.max(img)
    if i_max != 0:  # no 0 division
        img /= i_max

    return img


def imagej_metadata_tags(metadata, byteorder):
    """Return IJMetadata and IJMetadataByteCounts tags from metadata dict.

    The tags can be passed to the TiffWriter.save function as extratags.

    """
    header = [{'>': b'IJIJ', '<': b'JIJI'}[byteorder]]
    bytecounts = [0]
    body = []

    def writestring(data, byteorder):
        return data.encode('utf-16' + {'>': 'be', '<': 'le'}[byteorder])

    def writedoubles(data, byteorder):
        return struct.pack(byteorder + ('d' * len(data)), *data)

    def writebytes(data, byteorder):
        return data.tobytes()

    metadata_types = (
        ('Info', b'info', 1, writestring),
        ('Labels', b'labl', None, writestring),
        ('Ranges', b'rang', 1, writedoubles),
        ('LUTs', b'luts', None, writebytes),
        ('Plot', b'plot', 1, writebytes),
        ('ROI', b'roi ', 1, writebytes),
        ('Overlays', b'over', None, writebytes))

    for key, mtype, count, func in metadata_types:
        if key not in metadata:
            continue
        if byteorder == '<':
            mtype = mtype[::-1]
        values = metadata[key]
        if count is None:
            count = len(values)
        else:
            values = [values]
        header.append(mtype + struct.pack(byteorder + 'I', count))
        for value in values:
            data = func(value, byteorder)
            body.append(data)
            bytecounts.append(len(data))

    body = b''.join(body)
    header = b''.join(header)
    data = header + body
    bytecounts[0] = len(header)
    bytecounts = struct.pack(byteorder + ('I' * len(bytecounts)), *bytecounts)
    return ((50839, 'B', len(data), data, True),
            (50838, 'I', len(bytecounts) // 4, bytecounts, True))


# noinspection PyTypeChecker
def save_imagej_composite(filename, img_stack, color_order):
    """
    Saves an ImageJ composite image with given color order
    ----------
    filename : str
        path and filename to save image under
    img_stack : np.array
        image stack to save (must have 3 channels)
    color_order : list
        list of colors to apply to channels in img_stack
     """
    if img_stack.shape[1] != len(color_order):
        raise ValueError("Number of channels not equal to number of colors")

    val_range = np.arange(256, dtype=np.uint8)
    gray = np.stack([val_range, val_range, val_range])
    red = np.zeros((3, 256), dtype='uint8')
    red[0] = val_range
    green = np.zeros((3, 256), dtype='uint8')
    green[1] = val_range
    blue = np.zeros((3, 256), dtype='uint8')
    blue[2] = val_range
    magenta = np.zeros((3, 256), dtype='uint8')
    magenta[0] = val_range
    magenta[2] = val_range
    cyan = np.zeros((3, 256), dtype='uint8')
    cyan[1] = val_range
    cyan[2] = val_range
    yellow = np.zeros((3, 256), dtype='uint8')
    yellow[0] = val_range
    yellow[1] = val_range

    luts = []
    for col in color_order:
        if col in locals().keys():
            luts.append(locals()[col])
        else:
            raise ValueError(f"Unknown color {col}")

    ijtags = imagej_metadata_tags({'LUTs': luts}, '>')
    tifffile.imsave(
        filename,
        img_stack,
        metadata={'mode': 'composite'},
        byteorder='>',
        imagej=True,
        extratags=ijtags
    )


# argparse control
def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x <= 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError(f"{x} not greater than 0 and less than or equal to 1")
    return x


def positive_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a floating-point literal")

    if x <= 0.0:
        raise argparse.ArgumentTypeError(f"{x} not greater than 0")
    return x


def positive_int(x):
    try:
        x = int(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{x} not a int")

    if x <= 0:
        raise argparse.ArgumentTypeError(f"{x} not greater than 0")
    return x


# Segments based on connectivity
def fast_label_volume(input_volume):
    """
     Labels a volume based on connectivity
     ----------
     volume : np.array
         binary volume to label
     Returns
     -------
     np.array
         labeled volume, where 0=background
     int
        number of labels
     """
    return measure.label(input_volume, background=0, return_num=True)


# Segments the input volume using watershedding and returns labeled volume and number of labels
def label_volume(input_volume, min_distance=20):
    """
     Labels a volume using watershedding
     ----------
     volume : np.array
         binary volume to label
     min_distance : int
        minimum distance for peak finding
     Returns
     -------
     np.array
         labeled volume, where 0=background
     int
        number of labels
     """
    input_volume = input_volume.astype(int)
    distance = ndi.distance_transform_edt(input_volume)
    coords = peak_local_max(distance, min_distance=min_distance, footprint=np.ones((3, 3, 3)))  # magic number
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=input_volume)
    return labels, np.max(labels)




def r_from_vol(vol):
    return (vol / ((4 / 3) * math.pi)) ** (1 / 3)


def should_merge(vol1, vol2, shared_border, threshold=0.4):
    smallest_circ = math.pi * (min(r_from_vol(np.sum(vol1)), r_from_vol(np.sum(vol2))) ** 2)
    return smallest_circ * threshold < shared_border


def merge_connected_volumes(labeled_volume, area_ratio=0.3):
    """
    Merges volumes where their shared border is larger than the median cross section surface area * area_ratio threshold
    Parameters
    ----------
    labeled_volume
        Volume where a voxel value represents the label
    area_ratio
        Threshold for merging
    Returns
    -------
    np.array
        new labeled volume
    int
        number of labels found (not including background=0)
    """
    num_labels = np.max(labeled_volume)
    # create adjacency matrix
    adj_mat = np.zeros((num_labels + 1, num_labels + 1))
    filtered_vols = np.zeros((num_labels, *labeled_volume.shape))
    cur_vol = np.zeros_like(labeled_volume)
    prog = 0
    progress_bar(prog, num_labels)
    # check connectivity based on shared border

    # find connections
    for i in range(1, num_labels + 1):
        filtered_vols[i - 1] = (labeled_volume == i).astype(np.uint8) * i
        dilated = binary_dilation(filtered_vols[i - 1]).astype(np.uint8) * i
        cur_vol += dilated
        lab_max = np.max(cur_vol)
        if lab_max > i:
            for j in range(i + 1, lab_max + 1):
                other_idx = j - i
                # check if overlapping should be merged.
                # could be optimized by transitively closing adj matrix and checking whether it is already included.

                if j in cur_vol:
                    # print(f"{i} overlaps with {other_idx}")
                    if should_merge(filtered_vols[i - 1], filtered_vols[other_idx - 1], np.count_nonzero(cur_vol == j),
                                    threshold=area_ratio):
                        # print(f"merging {i} with {other_idx}")
                        adj_mat[i, other_idx] = 1
                        adj_mat[other_idx, i] = 1
        cur_vol = np.clip(cur_vol, 0, i)  # new overlap gets current label
        prog += 1
        progress_bar(prog, num_labels)

    # find connected components in adjacency matrix
    graph = csr_matrix(adj_mat)
    n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
    log_msg(f"reduced {num_labels} labels to {n_components - 1} components")
    # relabel the volume
    for i in range(num_labels + 1):
        labeled_volume[labeled_volume == i] = labels[i]

    return labeled_volume, n_components - 1


def get_shared_border(vol1, vol2):
    """
    Gets the surface area shared by two volumes
    Parameters
    ----------
    vol1 : np.array
        first volume
    vol2 : np.array
        second volume
    Returns
    -------
    int
        overlapping surface area
    """
    vol1_val = np.max(vol1)
    vol2_val = np.max(vol2)

    # force binary if not already the case
    if vol1_val != 1:
        vol1 = (vol1 > 0).astype(int)
    if vol2_val != 1:
        vol2 = (vol2 > 0).astype(int)

    dilated = binary_dilation(vol1)
    overlapped = dilated + vol2
    overlapping = np.count_nonzero(overlapped == 2)
    return overlapping
