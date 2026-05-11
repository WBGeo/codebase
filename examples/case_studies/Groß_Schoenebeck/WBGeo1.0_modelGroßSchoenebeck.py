# Importing necessary libraries
import pandas as pd
import os

from core.object_components import InputData_StructuralElements, InputData_FaultElements

from core.structural_modeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.structural_modeling_components.structural_modeling_visualization.structural_modeling_visualization import (plot_structural_model_2D, plot_structural_model_3D, plot_fault_model_2D, plot_fault_model_3D)

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
from core.meshing_components.mesh_format.exodus.Exo_format import export_mesh_results_to_exodus
from core.meshing_components.mesh_format.vtu.VTU_format import export_mesh_results_to_vtu
from core.meshing_components.mesh_format.vtk.VTK_format import export_mesh_results_to_vtk
from core.meshing_components.mesh_format.feflow.Feflow_format import export_mesh_results_to_feflow
from core.meshing_components.mesh_format.gmsh.GMSH_format import export_mesh_results_to_gmsh
from core.meshing_components.mesh_format.stl.STL_format import export_mesh_results_to_stl
from core.meshing_components.mesh_format.vtm.VTM_format import export_mesh_results_to_vtm
from core.meshing_components.mesh_format.ansys.Ansys_format import export_mesh_results_to_ansys
from core.meshing_components.mesh_format.abaqus.Abaqus_format import export_mesh_results_to_abaqus
#%%

cwd = os.getcwd()

#%%

surface_points = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_surface_points_downsampled.csv")
orientations = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_orientations_downsampled.csv")

surface_points_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_faults_surface_points_downsampled.csv")
orientations_faults = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/geological_data/schoenebeck_faults_orientations_downsampled.csv")

print(len(surface_points), len(orientations))
print(len(surface_points_faults), len(orientations_faults))


#%%

margin = 100  # Add a margin of 100 units around the data
grid = RegularGrid(
    extent=(surface_points["X"].min()-margin, surface_points["X"].max()+margin,
            surface_points["Y"].min()-margin, surface_points["Y"].max()+margin,
            surface_points["Z"].min()-margin, surface_points["Z"].max()+margin),
    resolution=(50, 50, 100)
)

#%%

# Create input data for the structural elements
data_elements = InputData_StructuralElements(name='GSB',
                                             mapping_object={
                                                "01": ('01_top_hannover'),
                                                "02": ('02_top_dethlingen'),
                                                "03": ('03_top_ebs'),
                                                "04": ('04_top_rockel'),
                                                "05": ('05_top_havel'),
                                                "06": ('06_top_vulkanit'),
                                                "07": ('07_top_karbon')
                                             },
                                             surface_points=surface_points,
                                             orientations=orientations
                                             )

# Create a StructuralFrame
frame = general.build_structural_frame(input_data_elements=data_elements,
                                       grid=grid
                                       )

frame.detailed_report()

#%%

# Plot the input input_data (2D and 3D possible)
plot_structural_model_2D(frame)
plot_structural_model_3D(frame)

#%%

# Set interpolation methods for each stratigraphic series
frame["01"].set_interpolation_method("Universal Co-Kriging")
frame["02"].set_interpolation_method("Universal Co-Kriging")
frame["03"].set_interpolation_method("Universal Co-Kriging")
frame["04"].set_interpolation_method("Universal Co-Kriging")
frame["05"].set_interpolation_method("Universal Co-Kriging")
frame["06"].set_interpolation_method("Universal Co-Kriging")
frame["07"].set_interpolation_method("Universal Co-Kriging")

frame.detailed_report()

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



###########################################################################################################################
#                                              Meshing
###########################################################################################################################
#################################
#  Explicit Structured meshing. #
#################################
mesh_str = create_structured_mesh_data(
    geomodel_result=structural_model_result,
    refinement_data=[10, 10, 5, 5, 6,7,9,10],
    mesh_division= (100, 100),
    z_threshold=0.1,
    tolerance=0.5
      )
#################################
# ImplicitStructured meshing.   #
#################################
mesh_implicit= create_implicit_structured_mesh(geomodel_result=structural_model_result)
#################################
# Explicit Unstructured meshing #
#################################
mesh_unstr = create_unstructured_mesh_data(
    geomodel_result=structural_model_result,
    tolerance=500,
    mesh_size=17,
    smooth=30,
    curve_mesh_size=5,
    extent=[401377.0, 409314.0, 5859433.0, 5865390.0, -3678.0, -4453.0]
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
#with open("filename_str_gross.exo", "wb") as f:
#    f.write(buf.getvalue())
# Implicit structured mesh
#buf = export_mesh_results_to_exodus(mesh_implicit)
#with open("filename_implic_gross.exo", "wb") as f:
#    f.write(buf.getvalue())
# Unstructured mesh
#buf = export_mesh_results_to_exodus(mesh_unstr)
#with open("filename_gross.exo", "wb") as f:
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

###############################################################
#                       Export Hierarchical meshes
# ##############################################################
#########################
# Export mesh to exodus #
#########################
#export_meshes_exodus(
#    mesh1_st,
#    mesh2_st,
#    closest_st,
#    mesh1_filename="big_mesh_st.exo",
#    mesh2_filename="small_mesh_st.exo",
#    closest_nodes_filename="closest_nodes_st.csv"
#)
#######################
# Export mesh to vtm  #
#######################
#export_meshes_vtm(
#    mesh1_u,
#    mesh2_u,
#    closest_u,
#    mesh1_filename="big_mesh_u.vtm.zip",
#    mesh2_filename="small_mesh_u.vtm.zip",
#    closest_nodes_filename="closest_nodes_u_vtm.csv"
#)
#######################
# Export mesh to vtu  #
#######################
#export_meshes_vtu(
#    mesh1_im,
#    mesh2_im,
#    closest_im,
#    mesh1_filename="big_mesh_im.vtu",
#    mesh2_filename="small_mesh.vtu",
#    closest_nodes_filename="closest_nodes_im_vtu.csv"
#)
#######################
# Export mesh to vtk  #
#######################
#export_meshes_vtk(
#    mesh1_u,
#    mesh2_u,
#    closest_u,
#    mesh1_filename="big_mesh_u.vtk",
#    mesh2_filename="small_mesh_u.vtk",
#    closest_nodes_filename="closest_nodes_u_vtk.csv"
#)
##########################
# Export mesh to feflow  #
##########################
#export_meshes_feflow(
#    mesh1_u,
#    mesh2_u,
#    closest_u,
#    mesh1_filename="big_mesh_u.fem",
#    mesh2_filename="small_mesh_u.fem",
#    closest_nodes_filename="closest_nodes_u_fem.csv"
#)
##########################
# Export mesh to abaqus  #
##########################
#export_meshes_abaqus(
#    mesh1_im,
#    mesh2_im,
#    closest_im,
#    mesh1_filename="big_mesh_im.inp",
#    mesh2_filename="small_mesh_im.inp",
#    closest_nodes_filename="closest_nodes_im_abaqus.csv"
#)
##########################
# Export mesh to ansys  #
##########################
#export_meshes_ansys(
#    mesh1_im,
#    mesh2_im,
#    closest_im,
#    mesh1_filename="big_mesh_im.mesh",
#    mesh2_filename="small_mesh_im.mesh",
#    closest_nodes_filename="closest_nodes_im_ansys.csv"
#)
########################
# Export mesh to gmsh  #
########################
#export_meshes_gmsh(
#    mesh1_u,
#    mesh2_u,
#    closest_u,
#    mesh1_filename="big_mesh_u.msh",
#    mesh2_filename="small_mesh_u.msh",
#    closest_nodes_filename="closest_nodes_u_gmsh.csv"
# )
