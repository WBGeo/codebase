# ------------------------
# create_structured_pkl.py
# ------------------------
import os
import pickle
import gzip
import pandas as pd

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.meshing_components.explicit.structured.mesh_data import (
    create_structured_mesh_data,
)

# ------------------------
# Wrapper (keep consistent with test style)
# ------------------------
class MeshResults:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements_structured = elements


# ------------------------
# Paths
# ------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

pkl_file = os.path.join(base_dir, "structured_mesh.pkl.gz")  # 👈 compressed file


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
    name="Model_1",
    mapping_object={"Strat_Series": ("rock2", "rock1")},
    surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
    orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv")),
)

# ------------------------
# Build model
# ------------------------
frame = general.build_structural_frame(
    input_data_elements=input_data,
    grid=grid
)

structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=False
)

# ------------------------
# Create structured mesh
# ------------------------
mesh_str = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=(10, 10, 10),
    mesh_devision=(30, 30),
    z_threshold=0.1,
    tolerance=1
)

print("[INFO] Structured mesh created")

# ------------------------
# Wrap result (IMPORTANT for test compatibility)
# ------------------------
mesh_result = MeshResults(
    nodes=mesh_str.nodes,
    elements=mesh_str.elements   # ✅ FIXED
)

# ------------------------
# SAVE AS GZIPPED PKL
# ------------------------
with gzip.open(pkl_file, "wb") as f:
    pickle.dump(mesh_result, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[SUCCESS] Saved structured mesh to: {pkl_file}")
print("[INFO] Nodes:", len(mesh_result.nodes))
print("[INFO] Element blocks:", len(mesh_result.elements_structured))
