import numpy as np


def element_list_from_dict(data_dict: dict, key_mask: list):
    """
    Create a single list of elements from a dictionary based on a boolean mask for the keys.
    """
    # Ensure all values are tuples
    for key in data_dict:
        if isinstance(data_dict[key], str):
            data_dict[key] = (data_dict[key],)

    # Boolean mask for the keys as a numpy array
    key_mask = np.array(key_mask)

    # Create a combined mask
    combined_mask = []
    keys = list(data_dict.keys())
    for i, key in enumerate(keys):
        combined_mask.extend([bool(key_mask[i])] * len(data_dict[key]))

    return combined_mask


# Normalize function
def normalize_vectors(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)  # Compute L2 norm
    return vectors / np.where(norms == 0, 1, norms)  # Avoid division by zero
