import numpy as np
import pyvista as pv
import warnings

from core.object_components import InputData_StructuralElements, MeshResults


#TODO: Adapt mesh plotting to new Structure
def plot_mesh_3d(mesh_results: MeshResults, input_data: InputData_StructuralElements, colors=None, style="surface",
                 show_plotter=True) -> pv.Plotter:
    """
    Plot the mesh for process simulation in 3D.

    Args:
        mesh_results (MeshResults): The mesh to plot.
        input_data (InputData_StructuralElements): The input input_data for the structural geological model.
        colors (Optional(list)): List of colors to use for the different formations
        style (str): The style of the mesh to plot.
    """
    # Suppress the specific warning about points not being a float type
    warnings.filterwarnings("ignore",
                            message="Points is not a float type. This can cause issues when transforming or applying "
                                    "filters. Casting to ``np.float32``. Disable this by passing "
                                    "``force_float=False``.")

    formations = input_data.surface_points['formation'].unique()

    if input_data.faults is not None:
        n_faults = np.sum(input_data.faults)
    else:
        n_faults = 0

    # Create a PyVista plotter
    plotter = pv.Plotter(off_screen=not show_plotter)

    # Set default colors
    if colors is None:
        colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

        colors = ['#673ab7', '#34a853', '#ea4335', '#fbbc05', '#4285f4',
                  '#c4e4fc', '#ffd4d4', '#fff4c2', '#c4f8bd',
                  '#f18d00', '#bbdaa4', '#a7cdf2', '#9bbff4', '#4a80f5']

    if input_data.faults is not None:

        data = input_data.mapping_object

        # Ensure all values are tuples
        for key in data:
            if isinstance(data[key], str):
                data[key] = (data[key],)

        fault_mask = []
        keys = list(data.keys())
        for i, key in enumerate(keys):
            fault_mask.extend([bool(input_data.faults[i])] * len(data[key]))

        fault_mask = ~np.array(fault_mask)

        masked_colors = [color for color, m in zip(colors, fault_mask) if m]
        masked_formations = [element for element, m in zip(formations, fault_mask) if m]

        # Add moose meshes to the plotter
        for i in range(len(formations) - np.sum(input_data.faults)):
            # plotter.add_mesh(mesh[0][i], show_edges=True, style='wireframe', color=masked_colors[i],
            #                  label=masked_formations[i])
            plotter.add_mesh(mesh_results.mesh[i], show_edges=True, style=style, color=masked_colors[i],
                             label=masked_formations[i])
    else:
        # Add moose meshes to the plotter
        for i in range(len(formations)):
            # plotter.add_mesh(mesh[0][i], show_edges=True, style='wireframe', color=colors[i],
            #                 label=formations[i])
            plotter.add_mesh(mesh_results.mesh[i], show_edges=True, style=style, color=colors[i],
                             label=formations[i])

    # Add basement with extra label
    basement_ID = int(len(formations) - n_faults)
    plotter.add_mesh(mesh_results.mesh[basement_ID], show_edges=True,
                     style=style,
                     color=colors[len(formations)])

    plotter.add_legend(size=(0.13, 0.13), loc='lower right', face='circle')

    # Set the bounds and grid of the plotter
    plotter.show_bounds(grid=True)

    # Set the camera position
    plotter.camera.view_angle = 30.0
    plotter.camera.azimuth = 25.0
    plotter.camera.elevation = -15.0

    if show_plotter:
        # Display the interactive plot
        plotter.show()

    return plotter
