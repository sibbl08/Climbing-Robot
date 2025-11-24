import coacd
import trimesh
import os

input_file = "Robot/assets/jug3_center.obj"
output_dir = "CoACD_Output"
os.makedirs(output_dir, exist_ok=True)

# Laden
mesh = trimesh.load(input_file, force="mesh", skip_materials=True)
mesh = coacd.Mesh(mesh.vertices, mesh.faces)

# Zerlegung
parts = coacd.run_coacd(mesh, threshold=0.1)

# Speichern
for i, part in enumerate(parts):
    # CoACD liefert in dieser Version Tupel (vertices, faces)
    vertices, faces = part
    trimesh.Trimesh(vertices=vertices, faces=faces).export(f"{output_dir}/part_{i}.obj")

print(f"Export erfolgreich: {len(parts)} Teile")