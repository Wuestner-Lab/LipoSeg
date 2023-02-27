import json
import pyvista as pv
f_manifest = open("manifest.json")
manifest = json.load(f_manifest)
p = pv.Plotter(lighting='three lights')
p.set_background('black', top='white')
print("Loading saved meshes...")
for mesh_data in manifest:
    mesh = pv.read(mesh_data['file'])
    p.add_mesh(mesh, color=mesh_data['color'], opacity=mesh_data['opacity'], diffuse=0.5, specular=0.5,
               ambient=0.5)

print("Opening PyVista rendering...")
p.show_bounds(grid='front', location='outer', all_edges=True, )
p.show_axes()
p.show()