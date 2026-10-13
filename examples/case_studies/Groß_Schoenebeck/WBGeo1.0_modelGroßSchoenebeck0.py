from pathlib import Path
# Importing necessary libraries
import pandas as pd
import numpy as np
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements
from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (
    plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)
from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import (
    create_unstructured_mesh_data,
    load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv,
    load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv)
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)
from core.meshing_components.explicit.unstructured.refinement_mesh import (
    Refinement, LinearWellRefinement, FunctionWellRefinement, EllipseRefinement,
    LinearSourceRefinement, FunctionSourceRefinement, TriangulationRefinement, FaultRefinement)
from core.simulation_components.simulation_packages.golem.golem_model_builder import (
    CustomGolemBuilder,
)

from core.simulation_components.simulation_packages.golem.golem_run import run_simulation_golem
from core.simulation_components.simulation_visualization.simulation_visualization import (
    plot_variable_at_a_time, plot_cross_section_2D, plot_builder_materials, plot_variable_time_series)
#%%

cwd = os.getcwd()



# Solving steady state of the problem using custom golem input file.
reservoir_input_path = (
    cwd
    + "/examples/case_studies/Groß_Schoenebeck/input_data/simulation_files/steady_full_t.i"
)

with open(reservoir_input_path, "r") as f:
    reservoir_input = f.read()

# Solving the problem with Golem
model_GS_Xindyu_transient_results = run_simulation_golem(
    problem=CustomGolemBuilder(
        input_file_contents=reservoir_input,
        mesh_file=cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "filename_Gross_Schoenebeck_SM.exo",
          print_linear_residuals=True,
    ),
    image_name="moose-dev:fixed",
    project_dir=os.environ["GOLEM_PROJECT_DIR"],
    docker_dir="/projects",
    output_dir=(
        cwd
        + "/examples/case_studies/Groß_Schoenebeck/"
          "output_steady_t/"
    ),
    nproc=4,
    stream_output=True,  # Show live GOLEM output (opt-in)
)
