import math

import numpy as np
import pyvista as pv
from skimage import measure


# This could probably be optimized using NumPy, although it is pretty cheap and rarely called
def get_biggest_area(vol):
    """
    Gets Z level with highest area in a 3D volume
    ----------
    vol : np.array
        the 3D binary volume to search
    Returns
    -------
    int
        Z index with highest area
    """
    biggest_idx = -1
    biggest = 0
    for i, v_slice in enumerate(vol):
        size = np.sum(v_slice)
        if size > biggest:
            biggest_idx = i
            biggest = size
    return biggest_idx


# returns X,Y indices of center of mass of given slice matrix
def get_center_of_mass(mat):
    """
    Gets center of mass coordinates for a 2D binary array
    ----------
    slice : np.array
        the input matrix
    Returns
    -------
    int, int
        the x,y position of the center of mass
    """
    return np.mean(np.argwhere(mat == 1), axis=0)


'''
Ideas for better circle generation

Find longest length inside area:
    Take center of mass, draw a line in each direction until it hits a 0 value. Rotate 360 degrees in some interval.
    
Fit a circle using hough circles:
    biggest valid circle wins
        


'''


# returns radius from assumed circular area (sqrt(area/pi))
def radius_from_area(area):
    """
    Gets the radius of a circle with given area
    ----------
    area : float
        area to calculate from
    Returns
    -------
    float
        radius of circle with given area
    """
    return math.sqrt(area / math.pi)


# takes a binary volume (0s and 1s) and returns a mesh of the object.
def mesh_from_volume(volume, force_sphere=False, scale=(1, 1)):
    """
     Generates a PyVista mesh from a binary volume using marching cubes or a sphere
     ----------
     volume : np.array
         binary volume to generate mesh from
     force_sphere : bool
        Generate sphere based on slice with highest area, if set to true
     scale : (float,float)
        x,y scaling to apply on mesh. May be applied if z-stack was resized in x,y dimension.
     Returns
     -------
     pv.DataSet
         Generated mesh
     """
    scale_x, scale_y = scale
    if force_sphere:
        z = get_biggest_area(volume)  # vol Z = rendering X
        x, y = get_center_of_mass(volume[z])
        x *= scale_x
        y *= scale_y
        # if scale is wrong, compensate when calculating radius
        r = radius_from_area(np.sum(volume[z]) * scale_x * scale_y)
        return pv.Sphere(radius=r, center=(y, x, z))
    else:
        # volume is loaded into PyVista in a weird way, so we swap axes to solve it.
        volume = np.swapaxes(volume, 0, 1)  # ZYX -> YZX
        volume = np.swapaxes(volume, 1, 2)  # YZX -> YXZ
        volume = np.swapaxes(volume, 0, 1)  # YXZ -> XYZ

        verts, faces, _, _ = measure.marching_cubes(volume, 0)
        face_list = []
        for subarr in faces:
            face_list.append(3)
            for val in subarr:
                face_list.append(int(val))

        surf = pv.PolyData(verts, face_list)
        surf = surf.transform(np.array([[scale_x, 0, 0, 0],
                                        [0, scale_y, 0, 0],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]]))
        return surf
