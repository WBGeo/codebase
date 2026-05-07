# Importing necessary libraries
import pandas as pd
import os

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.object_components import InputData_StructuralElements
from core.structural_modeling_components import general, general_faults
from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_3D, plot_input_data_3D)

from core.meshing_components.explicit.structured.mesh_data import create_structured_mesh_data
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data
from core.meshing_components.meshing_visualization.meshing_visualization import plot_mesh_3d
from core.meshing_components.implicit.export_implicit import create_implicit_structured_mesh
from core.meshing_components.explicit.unstructured.mesh_data import create_unstructured_mesh_data, \
  load_wells_from_csv, load_shafts_from_csv, load_sources_from_csv, load_planes_from_csv, load_ellipses_from_csv, load_triangulations_planes_from_csv
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.mesh_format.vtu.VTU_format import export_mesh_results_to_vtu
from core.meshing_components.mesh_format.vtk.VTK_format import export_mesh_results_to_vtk
from core.meshing_components.mesh_format.feflow.Feflow_format import export_mesh_results_to_feflow
from core.meshing_components.mesh_format.gmsh.GMSH_format import export_mesh_results_to_gmsh
from core.meshing_components.mesh_format.stl.STL_format import export_mesh_results_to_stl
from core.meshing_components.mesh_format.vtm.VTM_format import export_mesh_results_to_vtm
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
from core.meshing_components.mesh_format.abaqus.Abaqus_format import export_mesh_results_to_abaqus

from core.simulation_components.simulation_packages.Sfepy.simulation_run import run_sfepy
from core.simulation_components.output_format.VTK.unified_format_vtk import load_vtk_results
from core.simulation_components.visualisation.results_visualisation import plot_variable_at_a_time, plot_cross_section, plot_variable_along_line, print_variable_at_point, plot_variable_time_series

#%%

cwd = os.getcwd()

# WORKFLOW Synthetic Model 5: no faults, 1 unconformity, 2 stratigraphic groups

#%%

# Create grid
grid = RegularGrid(
    extent=(0, 2000, 0, 1000, 0, 1000),  # Example grid extent
    resolution=(100, 50, 50)  # Example resolution
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='Model_5',
                                             mapping_object={
                                                 "Strat_Series2": ('rock4', 'rock3'),
                                                 "Strat_Series1": ('rock2', 'rock1')},
                                             surface_points=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model5/input_data/geological_data/model5_surface_points_df.csv"),
                                             orientations=pd.read_csv(
                                                 cwd + "/examples/synthetic_examples/model5/input_data/geological_data/model5_orientations_df.csv")
                                             )

#%%

plot_input_data_3D(data_elements)

#%%

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid,
                                       )

frame.detailed_report()

#%%

# Plot the input data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame, show_surface_meshes=False)

#%%

# Set interpolation methods for each stratigraphic series

# Set another interpolation method per group
# frame["Strat_Series1"].set_interpolation_method("Universal Co-Kriging")
# frame["Strat_Series2"].set_interpolation_method("Universal Co-Kriging")

# Configure interpolation parameters if needed (available parameters depend on the interpolation method)
# frame["Strat_Series1"].configure_interpolation_params()
# frame["Strat_Series2"].configure_interpolation_params()

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

#%%

# Optional: Compute gradients at the surface mesh vertices
# from core.structural_modeling_components.structural_modeling_utility import surface_mesh_gradients
#
# gradients_dict, gradients_faults_dict = surface_mesh_gradients.get_surface_mesh_gradients(structural_model_result,
#                                                                                           mesh_type="unmasked")
# surface_mesh_gradients.plot_surface_mesh_gradients(structural_model_result,
#                                                    gradients_dict,
#                                                    gradients_faults_dict,
#                                                    mesh_type="unmasked")

#%%
###########################################################################################################################
#                                              Meshing
###########################################################################################################################
#################################
#  Explicit Structured meshing. #
#################################
#mesh_str = create_structured_mesh_data(
#    geomodel_result=structural_model_result,
#    refinement_data=[25, 21, 16, 5, 6],
#    z_threshold=0.1,
#    tolerance=1,
#    extent=[50, 2000, 50, 1000, 50, 1000]
# )
#################################
# ImplicitStructured meshing.   #
#################################
#mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result,extent=[1500, 2000, 500, 1000, 500, 1000])
#################################
# Explicit Unstructured meshing #
#################################
# load wells
wells = load_wells_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_wells.csv")
# load shafts
shafts=  load_shafts_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_shafts.csv")
# load point sources
sources= load_sources_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_sources.csv")
# load planes
planes= load_planes_from_csv(cwd + "/examples/synthetic_examples/model5/input_data/engineering_objects/model_5_planes.csv")


mesh_unstr = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    wells=wells,
    sources=sources,
    shafts= shafts,
    extra_planes= planes,
    tolerance=50,
    mesh_size=20,
    curve_mesh_size=10,
    DISTANCE_THRESHOLD = 50,
    PROJECTION_THRESHOLD = 60,
    EXTRUSION_FACTOR = 120,
    z_threshold = 10,
    extent=[50, 2000, 50, 1000, 50, 1000]
)

##########################################################################################################################
#                                                Plot the meshing results
##########################################################################################################################
#plot_mesh_3d(mesh_implicit, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_unstr, structural_model_result, show_plotter=True)
#plot_mesh_3d(mesh_str, structural_model_result, show_plotter=True)

###########################################################################################################################
#                                                    Exporting meshes
###########################################################################################################################
#########################
# Export mesh to exodus #
#########################
# Structured mesh
#buf = export_mesh_results_to_exodus(mesh_str)
#with open("filename_str1.exo", "wb") as f:
#    f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_exodus(mesh_implicit)
#with open("filename_implic2.exo", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_exodus(mesh_unstr)
#with open("filename.exo", "wb") as f:
#    f.write(buf.getvalue())
#######################
# Export mesh to vtu  #
#######################
# Structured mesh
#buf = export_mesh_results_to_vtu(mesh_str)
#with open("filename_str2.vtu", "wb") as f:
#    f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_vtu(mesh_implicit)
#with open("filename_implic2.vtu", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_vtu(mesh_unstr)
#with open("filename.vtu", "wb") as f:
#    f.write(buf.getvalue())
######################
# Export mesh to vtk #
######################
# Structured mesh
#buf = export_mesh_results_to_vtk(mesh_str)
#with open("filename_str2.vtk", "wb") as f:
#   f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_vtk(mesh_implicit)
#with open("filename_implic2.vtk", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_vtk(mesh_unstr)
#with open("filename.vtk", "wb") as f:
#    f.write(buf.getvalue())
#######################
# Export mesh to vtm  #
#######################
# Structured mesh
#buf = export_mesh_results_to_vtm(mesh_str)
#with open("filename_str2.vtm.zip", "wb") as f:
#    f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_vtm(mesh_implicit)
#with open("filename_implic2.vtm.zip", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_vtm(mesh_unstr)
#with open("filename.vtm.zip", "wb") as f:
#    f.write(buf.getvalue())
#########################################################
## Export mesh to stl (only unstructured is supported). #
#########################################################
# Unstructured mesh
#buf = export_mesh_results_to_stl(mesh_unstr)
#with open("filename.stl.zip", "wb") as f:
#    f.write(buf.getvalue())
#########################################################
# Export mesh to gmsh (only unstructured is supported)  #
#########################################################
# Unstructured mesh
#buf = export_mesh_results_to_gmsh(mesh_unstr)
#with open("filename.msh", "wb") as f:
#    f.write(buf.getvalue())
##########################################################
# Export mesh to feflow (only unstructured is supported) #
##########################################################
# Unstructured mesh
#buf = export_mesh_results_to_feflow(mesh_unstr)
#with open("filename.fem", "wb") as f:
#    f.write(buf.getvalue())
########################
# Export mesh to ansys #
########################
## Structured mesh
#buf = export_mesh_results_to_ansys(mesh_str)
#with open("filename_str2_ansys.msh", "wb") as f:
#    f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_vtm(mesh_implicit)
#with open("filename_implic2_ansys.msh", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_ansys(mesh_unstr)
#with open("filename_ansys.msh", "wb") as f:
#    f.write(buf.getvalue())
#########################
# Export mesh to abaqus #
#########################
# Unstructured mesh
#buf = export_mesh_results_to_abaqus(mesh_unstr)
#with open("filename.inp", "wb") as f:
#    f.write(buf.getvalue())
# Structured mesh
#buf = export_mesh_results_to_abaqus(mesh_str)
#with open("filename_st.inp", "wb") as f:
#    f.write(buf.getvalue())
# Implicit mesh
#buf = export_mesh_results_to_abaqus(mesh_implicit)
#with open("filename_imp.inp", "wb") as f:
#    f.write(buf.getvalue())


###########################################################################################################################
#                                                   simulation_components with Sfeepy
###########################################################################################################################
################################
# Explicit unstructured mesh.  #
################################
#mesh_unst = create_unstructured_mesh_data(
#    geomodel_result=structural_model_result,
#    mesh_size=50,
#    curve_mesh_size=5,
#)
#Sim_out=run_sfepy(cwd +'/examples/input_data/engineering_objects/Model1/Hydro_thermal.py', mesh_unst, 'results')
################################
# Explicit structured mesh.  #
################################
#mesh_str = create_structured_mesh_data(
#    geomodel_result=structural_model_result,
#    refinement_data=(40,40,40),
#    mesh_devision=(40,40),
#    z_threshold=0.1,
#    tolerance=1
#)
#Sim_out=run_sfepy(cwd +'/examples/input_data/engineering_objects/Model1/Hydro_thermal.py', mesh_str, 'results')
################################
# Implicit structured mesh.  #
################################
mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)
Sim_out=run_sfepy(cwd +'/examples/synthetic_examples/model5/input_data/simulation_input_file/Thermal.py', mesh_implicit, 'results')
###########################################################################################################################
#                                                   Visualization of simulation_components results
###########################################################################################################################
data_by_time=load_vtk_results(Sim_out)
for t in data_by_time.nodes_by_time:
    print(f"\n⏱ Time {t}")
    print("  Node data keys:", list(data_by_time.node_data_by_time[t].keys()))
    print("  Cell data keys:", list(data_by_time.cell_data_by_time[t].keys()))
plot_variable_at_a_time(data_by_time,"T", 0, cmap="coolwarm", scale=(1,1,1))
plot_cross_section(data_by_time,"T", 0, origin=(540,20,100), normal=(1,0,0))
plot_variable_along_line(data_by_time,"T", 0, p0=(500,20,50), p1=(500,20,1000))
print_variable_at_point(data_by_time,"T", 0, point=(500.0,20.0,500.0))
plot_variable_time_series(data_by_time,"T", point=(500.0,20.0,500.0))

# %%


