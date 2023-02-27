import json
import math

import numpy as np
import pyvista as pv

voxel_unit = "µm³"
surface_unit = "µm²"
distance_unit = "µm"
pixel_width = 0.0196


def to_human_dist(dist):
    return f"{round(dist * (pixel_width), 3)}{distance_unit}"

def to_human_area(area):
    return f"{round(area * (pixel_width**2), 3)}{area_unit}"

def to_human_vol(vol):
    return f"{round(vol * (pixel_width**3), 3)}{voxel_unit}"


def calc_dist_to_vac(lipid_mesh, vacuole_mesh, epsilon=0.5):
    """
    Approximates the distance from a lipid mesh to a vac mesh, by tracing between the centers of mass.
    May be inaccurate if either mesh is not close to a sphere.
    Parameters
    ----------
    lipid_mesh
        Lipid droplet mesh
    vacuole_mesh
        Vacuole mesh
    epsilon
        distances below this value are turned to 0
    Returns
    -------
    float
        Approximate distance between surfaces of each mesh.
    """
    vac_point, vac_cell = vacuole_mesh.ray_trace(vacuole_mesh.center, lipid_mesh.center, first_point=True)
    if len(vac_point) == 0:  # no hit means it is inside
        return 0  # inside
    lip_point, lip_cell = lipid_mesh.ray_trace(lipid_mesh.center, vacuole_mesh.center, first_point=True)
    if len(lip_point) == 0:  # no hit means it is inside
        return 0  # inside
    m2m = math.dist(lip_point, vac_point)  # mesh to mesh distance
    # if mesh center is closer to other mesh trace hit, there is an intersection, and we set the distance to 0
    if math.dist(lipid_mesh.center, vac_point) < math.dist(lipid_mesh.center, lip_point) or m2m < epsilon:
        return 0
    return round(m2m,3)


def calc_contact_area(lipid_mesh: pv.PolyData, vacuole_mesh: pv.PolyData):
    # create a copy of the mesh
    bigger = lipid_mesh.copy(deep=True)
    # scale it up and move it back to the same location
    og_center = bigger.center
    bigger = bigger.scale([1.1, 1.1, 1.1], inplace=True)
    to_move = np.array(og_center) - np.array(bigger.center)
    transformation = np.array([
        [1, 0, 0, to_move[0]],
        [0, 1, 0, to_move[1]],
        [0, 0, 1, to_move[2]],
        [0, 0, 0, 1]
    ])
    moved = bigger.transform(transformation)
    # get the points from the mesh and find the ones enclosed by the vacuole
    points = pv.PolyData(moved.points)
    selected = points.select_enclosed_points(vacuole_mesh, check_surface=False)["SelectedPoints"]
    extracted = points.extract_points(selected.view(bool))
    # if there are enough points, scale it back down and calculate the surface area.
    if len(extracted.points) > 5:
        rescaled = extracted.scale([0.91, 0.91, 0.91], inplace=True)
        surf = rescaled.delaunay_2d()
        return surf.area
    return 0


def analyze_manifest(folder, render=True):
    f_manifest = open(f"{folder}/manifest.json")
    manifest = json.load(f_manifest)
    lipids = []
    vacs = []
    p = pv.Plotter(lighting='three lights')
    p.set_background('black', top='white')
    analysis = open(f"{folder}/analysis.csv", "w")
    analysis.write(f"lipid_id,volume,surface_area,distance,consumed,contact_area\n")
    for mesh_data in manifest:
        mesh = pv.read(f"{folder}/{mesh_data['file']}")
        if mesh_data['color'] == "red":
            vacs.append(mesh)
        elif mesh_data['color'] == "green":
            lipids.append(mesh)
        else:
            p.add_mesh(mesh, color=mesh_data['color'], opacity=mesh_data['opacity'], diffuse=0.5, specular=0.5,
                       ambient=0.5)

    p.show_bounds(grid='front', location='outer', all_edges=True, )
    p.show_axes()

    if len(vacs) > 0:
        dist_mat = np.zeros((len(lipids), len(vacs)))
        inside_mat = np.zeros(len(lipids))
        for i, vac_mesh in enumerate(vacs):
            p.add_mesh(vac_mesh, color="red", opacity=0.4, diffuse=0.5, specular=0.5, ambient=0.5)
            for j, lip_mesh in enumerate(lipids):
                dist_mat[j, i] = math.dist(vac_mesh.center, lip_mesh.center)
                pv_point = pv.PolyData(lip_mesh.center)
                select = pv_point.select_enclosed_points(vac_mesh, check_surface=False)
                inside_mat[j] = max(inside_mat[j], select["SelectedPoints"][0])

        for i, lip_mesh in enumerate(lipids):
            if inside_mat[i]:
                col = "blue"
            else:
                col = "green"
            p.add_mesh(lip_mesh, color=col, opacity=0.8, diffuse=0.5, specular=0.5, ambient=0.5)

            closest_vac = vacs[np.argmin(dist_mat[i])]
            distance = calc_dist_to_vac(lip_mesh, closest_vac)  # get idx with lowest dist to center
            contact_area = calc_contact_area(lip_mesh, closest_vac)
            p.add_point_labels([lip_mesh.center], [f"Lipid {i}\n"
                                                   f"{to_human_dist(distance)}\n"
                                                   f"{to_human_vol(lip_mesh.volume)}\n"
                                                   f"{np.around((contact_area/lip_mesh.area)*100,2)}%"])
            analysis.write(
                f"lipid {i},{lip_mesh.volume},{lip_mesh.area},{distance},{inside_mat[i]},{contact_area}\n")
    analysis.close()
    if render:
        p.show()


if __name__ == "__main__":
    fol = "."
    analyze_manifest(fol)
