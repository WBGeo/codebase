# ------------------------
# save_generated_mesh.py (OPTIMIZED + COMPRESSED)
# ------------------------

import pickle
import os
import pandas as pd
import numpy as np
import gzip

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
    load_wells_from_csv,
    load_shafts_from_csv,
    load_sources_from_csv,
    load_planes_from_csv,
    load_ellipses_from_csv,
    load_triangulations_planes_from_csv,
)

# ------------------------
# Minimal wrapper (same as test)
# ------------------------
class MeshResults:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements_unstructured = elements


# ------------------------
# Paths
# ------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

engineering_dir = os.path.join(data_dir, "../Engineering_objects/")

pkl_file = os.path.join(base_dir, "mesh_test.pkl.gz")


# ------------------------
# Grid
# ------------------------
grid = RegularGrid(
    extent=(0, 1000, 0, 1000, 0, 1000),
    resolution=(50, 50, 50)
)


# ------------------------
# Input data
# ------------------------
input_data = InputData_StructuralElements(
    name='Model_1',
    mapping_object={"Strat_Series": ('rock2', 'rock1')},
    surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
    orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv"))
)


# ------------------------
# Build structural model
# ------------------------
frame = general.build_structural_frame(
    input_data_elements=input_data,
    grid=grid
)

geomodel_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=False
)


# ------------------------
# Engineering objects
# ------------------------
wells = load_wells_from_csv(os.path.join(engineering_dir, "model_1_wells.csv"))
shafts = load_shafts_from_csv(os.path.join(engineering_dir, "model_1_shafts.csv"))
sources = load_sources_from_csv(os.path.join(engineering_dir, "model_1_sources.csv"))
planes = load_planes_from_csv(os.path.join(engineering_dir, "model_1_planes.csv"))
ellipses = load_ellipses_from_csv(os.path.join(engineering_dir, "model_1_ellipses.csv"))
triangulations = load_triangulations_planes_from_csv(
    os.path.join(engineering_dir, "seismic_plane_new_offset.csv")
)


# ------------------------
# Create mesh
# ------------------------
mesh_generated = create_unstructured_mesh_data(
    geomodel_result=geomodel_result,
    wells=wells,
    sources=sources,
    shafts=shafts,
    triangulations=triangulations,
    extra_planes=planes,
    ellipses=ellipses,
    mesh_size=75,
    curve_mesh_size=5
)

print("[INFO] Mesh generation finished!")


# ------------------------
# Wrap result
# ------------------------
mesh_result = MeshResults(
    nodes=mesh_generated.nodes,
    elements=mesh_generated.elements
)


# ------------------------
# 🔥 OPTIMIZATION STEP 1: reduce precision
# ------------------------
mesh_result.nodes = mesh_result.nodes.astype(np.float32)

for block in mesh_result.elements_unstructured:
    block.data = block.data.astype(np.int32)


# ------------------------
# Save compressed PKL (gzip + best protocol)
# ------------------------
with gzip.open(pkl_file, "wb") as f:
    pickle.dump(mesh_result, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------
# Info
# ------------------------
print(f"[SUCCESS] Compressed mesh saved: {pkl_file}")
print(f"[INFO] Node count: {len(mesh_result.nodes)}")
print(f"[INFO] Element blocks: {len(mesh_result.elements_unstructured)}")
