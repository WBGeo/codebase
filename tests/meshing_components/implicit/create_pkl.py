# ------------------------
# create_implicit_pkl.py
# ------------------------
import os
import pickle
import gzip
import pandas as pd

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.meshing_components.implicit.export_implicit import (
    create_implicit_structured_mesh,
)

# ------------------------
# Wrapper (consistent with test)
# ------------------------
class MeshResults:
    def __init__(self, nodes, elements):
        self.nodes = nodes
        self.elements_implicit = elements


# ------------------------
# Paths
# ------------------------
base_dir = os.path.dirname(__file__)

data_dir = os.path.join(
    base_dir,
    "../../../examples/synthetic_examples/Model1/input_data/Geological_data/"
)

pkl_file = os.path.join(base_dir, "implicit_mesh.pkl.gz")


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
    mapping_object={"Strat_Series1": ("rock2", "rock1")},
    surface_points=pd.read_csv(os.path.join(data_dir, "model1_surface_points_df.csv")),
    orientations=pd.read_csv(os.path.join(data_dir, "model1_orientations_df.csv")),
)

# ------------------------
# Build structural model
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
# Create implicit structured mesh
# ------------------------
mesh_implicit = create_implicit_structured_mesh(
    geomodel_result=structural_model_result
)

print("[INFO] Implicit mesh created")

# ------------------------
# Wrap result
# ------------------------
# ⚠️ IMPORTANT: adjust attribute names if needed
nodes = getattr(mesh_implicit, "nodes", None)
elements = getattr(mesh_implicit, "elements", None)

if nodes is None or elements is None:
    raise AttributeError(
        "mesh_implicit does not have expected attributes 'nodes' and 'elements'. "
        "Print dir(mesh_implicit) to inspect."
    )

mesh_result = MeshResults(
    nodes=nodes,
    elements=elements
)

# ------------------------
# Save (compressed PKL)
# ------------------------
with gzip.open(pkl_file, "wb") as f:
    pickle.dump(mesh_result, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"[SUCCESS] Saved implicit mesh to: {pkl_file}")
print("[INFO] Nodes:", len(mesh_result.nodes))
print("[INFO] Element blocks:", len(mesh_result.elements_implicit))
