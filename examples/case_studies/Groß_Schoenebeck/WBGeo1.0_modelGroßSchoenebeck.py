# Importing necessary libraries
import numpy as np
import pandas as pd
import os

from core.structuralmodeling_components import general

from core.structuralmodeling_components.structural_objects.grids.grid_classes import RegularGrid

from core.visualization_components import (plot_structural_model_2D, plot_structural_model_3D,
                                           plot_fault_frame_3D)

#%%

cwd = os.getcwd()

#%%

# load the csv
df = pd.read_csv(cwd+"/examples/case_studies/Groß_Schoenebeck/input_data/schoenebeck_surface_points.csv")

#%%

# Downsample the data for faster computation
def spatial_downsample_by_formation(
    df: pd.DataFrame,
    percentage: float,
    n_bins: int = 50,
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Spatially downsample contact points per formation while preserving XY coverage.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns ['X', 'Y', 'Z', 'formation']
    percentage : float
        Fraction of points to keep (0 < percentage <= 1)
    n_bins : int
        Number of bins per axis for XY stratification
    random_state : int | None
        Seed for reproducibility

    Returns
    -------
    pd.DataFrame
        Downsampled dataframe
    """
    rng = np.random.default_rng(random_state)
    out = []

    for formation, g in df.groupby("formation"):
        n_target = int(len(g) * percentage)
        if n_target == 0:
            continue

        # Compute 2D bin indices
        x_bins = np.linspace(g["X"].min(), g["X"].max(), n_bins + 1)
        y_bins = np.linspace(g["Y"].min(), g["Y"].max(), n_bins + 1)

        x_idx = np.digitize(g["X"], x_bins) - 1
        y_idx = np.digitize(g["Y"], y_bins) - 1

        g = g.copy()
        g["_bin"] = list(zip(x_idx, y_idx))

        # Group by spatial bin
        bin_groups = g.groupby("_bin")

        # Target per bin (at least 1 if possible)
        n_bins_nonempty = len(bin_groups)
        per_bin = max(1, n_target // n_bins_nonempty)

        samples = []
        for _, bg in bin_groups:
            if len(bg) <= per_bin:
                samples.append(bg)
            else:
                samples.append(bg.sample(per_bin, random_state=random_state))

        sampled = pd.concat(samples)

        # If we overshot or undershot slightly, fix it
        if len(sampled) > n_target:
            sampled = sampled.sample(n_target, random_state=random_state)
        elif len(sampled) < n_target:
            remaining = g.drop(sampled.index)
            needed = n_target - len(sampled)
            if needed > 0 and len(remaining) > 0:
                sampled = pd.concat([
                    sampled,
                    remaining.sample(min(needed, len(remaining)), random_state=random_state)
                ])

        out.append(sampled.drop(columns="_bin"))

    return pd.concat(out, ignore_index=True)

#%%

df_ds = spatial_downsample_by_formation(
    df,
    percentage=0.025,   # keep 20%
    n_bins=40,
    random_state=42,
)

#%%

df_ds


#%%

grid = RegularGrid(
    extent=(df["X"].min(), df["X"].max(), df["Y"].min(), df["Y"].max(), df["Z"].min(), df["Z"].max()),
    resolution=(50, 50, 50)
)

#%%

# Create a StructuralFrame
frame = general.build_structural_frame({
                                        "Main": ('01_top_hannover',
                                                 '02_top_dethlingen',
                                                 '03_top_ebs',
                                                 '04_top_rockel',
                                                 '05_top_havel',
                                                 '06_top_vulkanit',
                                                 '07_top_karbon')
                                       },
                                        grid,
                                        df_ds)

frame.detailed_report()

#%%

plot_structural_model_3D(frame=frame, show_surface_meshes=False)

#%%

# # OK
# frame["Main"].set_interpolation_method("Ordinary Kriging")
# frame["Main"].configure_interpolation_params(range=5000, anisotropy_scaling_z=0.01, variogram_model="spherical")

# RBF
# frame["Main"].set_interpolation_method("Radial Basis Function")
# frame["Main"].configure_interpolation_params(kernel="multiquadric", epsilon=0.0001)


#%%

general.compute_structural_model(
    frame,
    fault_frame=None,  # or None for single-domain
    extract_meshes=True,
    verbose=True,
)

#%%

plot_structural_model_3D(frame=frame, show_surface_meshes=True)