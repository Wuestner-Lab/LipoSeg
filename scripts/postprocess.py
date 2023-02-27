import cv2
import numpy as np
from matplotlib.path import Path
from scipy.spatial import ConvexHull

kernel = np.ones((5, 5), np.uint8)


def erode(img, its=1):
    return cv2.erode(img, kernel, iterations=its)


def dilate(img, its=1):
    return cv2.dilate(img, kernel, iterations=its)


def create_mask(stack, alpha=0.05):
    """
    Generates a mask from a stack by z-projecting and thresholding at the given alpha and eroding and dilating
    ----------
    stack : np.array
        the stack to generate the mask from
    alpha : np.array
        the lower threshold bound
    Returns
    -------
    np.array
        a mask
    """
    # get mean image
    mean_img = np.mean(stack, 0)

    # threshold
    mask = (mean_img > alpha).astype(np.uint8)

    # erode twice to remove noise
    mask = erode(mask, 2)

    # dilate 5 times to fill holes and ensure signal is not masked off in the future.
    mask = dilate(mask, 5)

    return mask


def apply_mask(stack, mask):
    """
    Applies mask to each slice in the stack
    ----------
    stack : np.array
        the source stack for masking
    mask : np.array
        the mask to apply on the source
    Returns
    -------
    np.array
        masked stack
    """
    new_stack = np.zeros_like(stack)
    for i in range(len(stack)):
        new_stack[i] = stack[i] * mask

    return new_stack


# for masks with paired frames rather than a global mask.
def apply_mask_stack(stack, mask, threshold=0.1):
    """
    Applies mask to a stack with same shape pairwise
    ----------
    stack : np.array
        the source stack for masking
    mask : np.array
        the mask stack to apply on the source
    Returns
    -------
    np.array
        masked stack
    """
    new_stack = np.zeros_like(stack)
    mask = (mask > threshold).astype(np.uint8)  # force all values over 0 to become 1.
    for i in range(len(stack)):
        new_stack[i] = stack[i] * mask[i]

    return new_stack


def ema_stack(stack, alpha=0.25):
    """
    Applies exponential moving average over slices in a Z stack (first dimension of multidimensional array)
    ----------
    stack : np.array
        the source stack for masking
    alpha : float
        weight of the current slice at each time step
    Returns
    -------
    np.array
        processed stack
    """
    smoothed = np.zeros_like(stack)
    smoothed[0] = stack[0]  # We may start in the middle of a cell, so we have to just hope for the best here.
    for i in range(1, len(stack)):  # have to skip first one, since it is an edge case.
        smoothed[i] = stack[i] * alpha + smoothed[i - 1] * (1 - alpha)

    return smoothed

def force_convex(stack):
    """
    Generates and fills a convex hull from slices containing non-zero values in a 3D array.
    Only works for stacks containing a single object.
    ----------
    stack : np.array
        the input stack to be forced convex
    Returns
    -------
    np.array
        processed convex stack
    """
    out = np.zeros_like(stack)
    for i in range(len(stack)):

        binary = (stack[i] > 0).astype(np.uint8)
        points = np.argwhere(binary > 0)
        if len(points) > 3:
            hull = ConvexHull(points=points)

            vert_points = hull.points[hull.vertices].astype(np.uint8)
            path = Path(vert_points)

            coords = np.argwhere(np.ones_like(binary) == 1)

            contained = path.contains_points(coords)
            out[i] = contained.reshape((512, 512)) * np.ones_like(binary)

    return out
