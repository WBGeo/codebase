# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)

from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components import general, general_faults
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D)

from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.mesh_format.mesh_export import (
    export_mesh_results_to_exodus, export_mesh_results_to_vtu,
    export_mesh_results_to_vtk, export_mesh_results_to_feflow,
    export_mesh_results_to_gmsh, export_mesh_results_to_stl,
    export_mesh_results_to_vtm, export_mesh_results_to_ansys,
    export_mesh_results_to_abaqus)

#%%

cwd = os.getcwd()

#%%

# Create a grid for the model
grid = RegularGrid(
    extent=(5623500, 5640000, 32304500, 32305500, -3000, 500),  # Example grid extent
    resolution=(250, 20, 125)  # Example resolution
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_9',
                                             mapping_object={
                                                "Strat_Series1":
                                                    ('BreitgangFM',
                                                     'KrebsTraufeFM',
                                                     'WilhelmineFM',
                                                     'ObererKohlenkalkGP',
                                                     'MittlererKohlenkalkGP',
                                                     'CondrozGP')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/case_studies/Weisweiler/input_data/geological_data/modelWeisweilerMini_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/case_studies/Weisweiler/input_data/geological_data/modelWeisweilerMini_orientations_df.csv")
                                             )

# Create a StructuralFrame and include the fault frame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid)

frame.detailed_report()


#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()

# frame.detailed_report()

#%%

# Compute structural model result
structural_model_result = general.compute_structural_model(
    frame,
    extract_meshes=True,
    verbose=True,
)

#%%

# Plot the results (2D and 3D possible)
plot_structural_model_2D(structural_model_result.structural_frame)
plot_structural_model_3D(structural_model_result.structural_frame, show_surface_meshes=True)

#%%

# Optional plotting
# frame.plot_scalar_field_section(group_nr=0, axis='y', index=12)
# frame.plot_age_mask_section(group_nr=0, axis='y', index=12)


# Creare mesh
# Implicit structured mesh (without engineering objects)
mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)

# Implicit unstructured mesh (without engineering objects)
mesh_explicit_unstr = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=700,
    mesh_size=50,
    smooth=0.02,
    curve_mesh_size=8,
    mapping_litho='none',

)

# Explicit structured mesh
# NOTE: Currently structured mesh does not support models whose layers do not extend the extent of model

# Plot the meshing results
plot_mesh_3d(mesh_implicit, structural_model_result, show_plotter=True)
plot_mesh_3d(mesh_explicit_unstr, structural_model_result, show_plotter=True)

#%%

# Optional: Example of how to export the unstructured mesh to Exodus format.
# Similar functions are available for other formats (VTU, VTK, FEFLOW, GMSH, STL, VTM, Ansys, Abaqus).
# NOTE: Only the Exodus exporter requires a type specification (e.g. 'imp', 'str', 'unstr').
# In addition, some export formats support only specific mesh types.
# See the meshing manual/documentation for details.
# buf = export_mesh_results_to_exodus(mesh_explicit_unstr, type='unstr')
# with open("filename_Gross_Schoenebeck.exo", "wb") as f:
#    f.write(buf.getvalue())
