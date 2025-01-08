import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
def plot_surfaces(bottom_top_surfaces, n_gx, n_gy):

    plotter = pv.Plotter()

    # Extract x, y, z values for bottom surface (first row)
    surface_1_x = bottom_top_surfaces[0, :n_gx * n_gy]
    surface_1_y = bottom_top_surfaces[0, n_gx * n_gy: 2 * n_gx * n_gy]
    surface_1_z = bottom_top_surfaces[0, 2 * n_gx * n_gy: 3 * n_gx * n_gy]

    # Extract x, y, z values for top surface (second row)
    surface_2_x = bottom_top_surfaces[1, :n_gx * n_gy]
    surface_2_y = bottom_top_surfaces[1, n_gx * n_gy: 2 * n_gx * n_gy]
    surface_2_z = bottom_top_surfaces[1, 2 * n_gx * n_gy:]

    # Create mesh for surface 1 (bottom surface)
    surface_1 = np.column_stack((surface_1_x, surface_1_y, surface_1_z))
    surface_1_mesh = pv.PolyData(surface_1)

    # Create mesh for surface 2 (top surface)
    surface_2 = np.column_stack((surface_2_x, surface_2_y, surface_2_z))
    surface_2_mesh = pv.PolyData(surface_2)

    # Add the surfaces to the plotter with colors
    plotter.add_mesh(surface_1_mesh, color='red', label='Surface 1 (Bottom)')
    plotter.add_mesh(surface_2_mesh, color='blue', label='Surface 2 (Top)')
    plotter.add_axes()

    # Add a legend
    plotter.add_legend()


    plotter.show()

def plot_points_with_ids(all_points_array, n_gx, n_gy):
    """
    Plots points from all_points_array, ensuring that points with the same ID have the same color.

    Args:
        all_points_array (np.array): Array of shape (n_rows, 3 * n_gx * n_gy + 1), where each row contains:
                                      - x values (first n_gx * n_gy entries)
                                      - y values (next n_gx * n_gy entries)
                                      - z values (next n_gx * n_gy entries)
                                      - ID values (last entry)
        n_gx (int): Grid size in x direction.
        n_gy (int): Grid size in y direction.
    """
    # Extract x, y, z values and ids from the array
    x = all_points_array[:, :n_gx * n_gy]
    y = all_points_array[:, n_gx * n_gy: 2 * n_gx * n_gy]
    z = all_points_array[:, 2 * n_gx * n_gy: 3 * n_gx * n_gy]
    ids = all_points_array[:, -n_gx * n_gy:]

    # Initialize a PyVista plotter
    plotter = pv.Plotter()

    # Get unique IDs
    unique_ids = np.unique(ids)
    legend_entries = []

    # Generate a color map (using the 'tab10' colormap here)
    cmap = plt.cm.get_cmap('tab10', len(unique_ids))

    # Plot each group of points with the same ID
    for i, unique_id in enumerate(unique_ids):
        # Mask the points with the current ID
        mask = ids == unique_id
        x_masked = x[mask]
        y_masked = y[mask]
        z_masked = z[mask]

        # Stack x, y, and z values correctly
        points = np.column_stack([x_masked.flatten(), y_masked.flatten(), z_masked.flatten()])

        # Define color for the current group (from the colormap)
        color = mcolors.rgb2hex(cmap(i)[:3])  # Convert RGB to hex

        # Plot the points
        plotter.add_mesh(pv.PolyData(points), color=color, point_size=5, render_points_as_spheres=True)
        legend_entries.append([f"ID: {unique_id}", color])
    plotter.add_legend(legend_entries, bcolor='white')
    plotter.add_axes()

    # Show the plot
    plotter.show()




