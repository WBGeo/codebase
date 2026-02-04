# Importing necessary libraries
import numpy as np
import pandas as pd
import os
from scipy.spatial import Delaunay

#%%

cwd = os.getcwd()

#%%

# load the csv
# This file too big for gitlab
df = pd.read_csv("C:/Users/vonha/Desktop/schoenebeck_surface_points.csv")
df.describe()

df_faults_1 = pd.read_csv("C:/Users/vonha/Desktop/water.csv")
df_faults_2 = pd.read_csv("C:/Users/vonha/Desktop/seismic_plane.csv")

# Combine fault data and add formation column with name of input file
df_faults_1['formation'] = 'water'
df_faults_2['formation'] = 'seismic_plane'
df_faults = pd.concat([df_faults_1, df_faults_2], ignore_index=True)

df_faults.head()

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

def compute_surface_orientations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for formation, g in df.groupby("formation"):
        if len(g) < 3:
            continue

        pts = g[["X", "Y", "Z"]].to_numpy()

        # --- dominant normal via PCA ---
        coords = pts - pts.mean(axis=0)
        _, _, vh = np.linalg.svd(coords, full_matrices=False)
        dominant_normal = vh[-1]

        # Optional global sign convention
        if dominant_normal[2] < 0:
            dominant_normal = -dominant_normal

        tri = Delaunay(pts[:, :2])

        for simplex in tri.simplices:
            p0, p1, p2 = pts[simplex]

            v1 = p1 - p0
            v2 = p2 - p0

            n = np.cross(v1, v2)
            norm = np.linalg.norm(n)
            if norm == 0:
                continue

            n /= norm

            # Enforce consistency
            if np.dot(n, dominant_normal) < 0:
                n = -n

            c = (p0 + p1 + p2) / 3

            rows.append({
                "X": c[0],
                "Y": c[1],
                "Z": c[2],
                "G_x": n[0],
                "G_y": n[1],
                "G_z": n[2],
                "formation": formation,
            })

    return pd.DataFrame(rows)



def spatially_downsample_orientations(
    df: pd.DataFrame,
    n_target: int = 25,
    n_bins: int = 6,
    margin_frac: float = 0.1,   # NEW: fraction of extent to exclude per side
    random_state: int | None = None,
) -> pd.DataFrame:
    """
    Spatially downsample orientation vectors per formation,
    excluding a margin around the spatial extent.
    """
    rng = np.random.default_rng(random_state)
    out = []

    for formation, g in df.groupby("formation"):
        if len(g) <= n_target:
            out.append(g)
            continue

        # --- compute interior bounds ---
        x_min, x_max = g["X"].min(), g["X"].max()
        y_min, y_max = g["Y"].min(), g["Y"].max()

        x_margin = margin_frac * (x_max - x_min)
        y_margin = margin_frac * (y_max - y_min)

        interior = g[
            (g["X"] >= x_min + x_margin) &
            (g["X"] <= x_max - x_margin) &
            (g["Y"] >= y_min + y_margin) &
            (g["Y"] <= y_max - y_margin)
        ]

        # If margin removes too much data, fall back
        if len(interior) < n_target:
            interior = g

        g = interior

        # --- binning ---
        x_bins = np.linspace(g["X"].min(), g["X"].max(), n_bins + 1)
        y_bins = np.linspace(g["Y"].min(), g["Y"].max(), n_bins + 1)

        xi = np.digitize(g["X"], x_bins) - 1
        yi = np.digitize(g["Y"], y_bins) - 1

        g = g.copy()
        g["_bin"] = list(zip(xi, yi))

        bin_groups = g.groupby("_bin")
        per_bin = max(1, n_target // len(bin_groups))

        samples = []
        for _, bg in bin_groups:
            if len(bg) <= per_bin:
                samples.append(bg)
            else:
                samples.append(bg.sample(per_bin, random_state=random_state))

        sampled = pd.concat(samples)

        # --- final adjustment ---
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

# Downsample surface points to 1% per formation
downsampled_surface_df = spatial_downsample_by_formation(
    df,
    percentage=0.005,   # 0.5%
    n_bins=40,
    random_state=42,
)

# Save to csv
downsampled_surface_df.to_csv(os.path.join(cwd, "examples/case_studies/Groß_Schoenebeck/input_data"
                                                "/schoenebeck_surface_points_downsampled.csv"), index=False)

#%%

# Step 1: compute normals
orientations = compute_surface_orientations(downsampled_surface_df)

# Step 2: reduce to ~25 per formation
orientations_ds = spatially_downsample_orientations(
    orientations,
    n_target=10,
    n_bins=6,
    random_state=42,
)

# Save to csv
orientations_ds.to_csv(os.path.join(cwd, "examples/case_studies/Groß_Schoenebeck/input_data"
                                        "/schoenebeck_orientations_downsampled.csv"), index=False)

#%%

# Same for faults

# Downsample surface points to 1% per formation
downsampled_surface_df_faults = spatial_downsample_by_formation(
    df_faults,
    percentage=0.1,   # 10%
    n_bins=40,
    random_state=42,
)

# Save to csv
downsampled_surface_df_faults.to_csv(os.path.join(cwd, "examples/case_studies/Groß_Schoenebeck/input_data"
                                                "/schoenebeck_faults_surface_points_downsampled.csv"), index=False)

#%%

# Step 1: compute normals
orientations_faults = compute_surface_orientations(downsampled_surface_df_faults)

# Step 2: reduce to ~25 per formation
orientations_ds_faults = spatially_downsample_orientations(
    orientations_faults,
    n_target=10,
    n_bins=6,
    random_state=42,
)

# Save to csv
orientations_ds_faults.to_csv(os.path.join(cwd, "examples/case_studies/Groß_Schoenebeck/input_data"
                                        "/schoenebeck_faults_orientations_downsampled.csv"), index=False)