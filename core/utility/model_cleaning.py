import numpy as np
from scipy.ndimage import label, generate_binary_structure


def remove_outliers_3d(rock_array):
    """
    Removes single voxel outliers in a 3D rock layer array by replacing them with
    the most frequent value from their 6 neighboring cells.

    Parameters:
        rock_array (ndarray): 3D numpy array containing rock unit labels.

    Returns:
        ndarray: Cleaned rock layer array.
    """
    cleaned_array = rock_array.copy()  # Copy to avoid modifying the original data
    unique_values = np.unique(rock_array)  # Get unique rock unit values

    # Define 6-connectivity for 3D neighbor detection (no diagonal neighbors)
    connectivity = generate_binary_structure(rank=3, connectivity=1)

    for value in unique_values:
        # Create a binary mask for the current rock unit
        mask = (rock_array == value)

        # Label connected components
        labeled_array, num_features = label(mask, structure=connectivity)

        # Count the size of each connected component
        sizes = np.bincount(labeled_array.ravel())
        sizes[0] = 0  # Ignore background (label 0)

        # Identify small components (single voxel outliers)
        small_labels = np.where(sizes == 1)[0]  # Small components are single voxels

        # Mask for outliers (single voxel)
        outlier_mask = np.isin(labeled_array, small_labels) & (labeled_array > 0)

        print(f"Removing {np.sum(outlier_mask)} outlier voxels for value {value}")

        # Replace outliers by checking the 6 neighbors
        if np.any(outlier_mask):
            # Iterate over the outlier voxels and replace them with the most frequent neighbor value
            for z, y, x in zip(*np.where(outlier_mask)):
                # Get the 6 neighbors
                neighbors = rock_array[max(0, z - 1):z + 2, max(0, y - 1):y + 2, max(0, x - 1):x + 2]

                # Exclude the current voxel (itself) from the neighbors
                neighbors = neighbors[neighbors != rock_array[z, y, x]]

                # Get the most frequent neighbor value
                if neighbors.size > 0:
                    most_frequent = np.bincount(neighbors.ravel()).argmax()
                    cleaned_array[z, y, x] = most_frequent
                else:
                    # If no valid neighbors (unlikely), retain the original value
                    cleaned_array[z, y, x] = value

    return cleaned_array



