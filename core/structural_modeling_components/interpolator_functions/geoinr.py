"""
GeoINR interpolation for a single structural group.

This module contains:
- `interpolate_group_geo_inr`: your framework-facing interpolator (pure function)
- A small PyTorch MLP (`ConcatMLP`) and helper functions used by GeoINR training/inference
- `stratigraphic_ConcatMLP`: training loop producing predictions and iso-values

Important notes
---------------
- `loss_intf` and `loss_grad` are defined twice in the original source. This is likely accidental,
  but removing one definition would be a code change, so both are kept. The second definition
  overwrites the first at runtime (standard Python behavior).
"""

from __future__ import annotations

import warnings
import time
from typing import Any, Dict, Optional, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyvista as pv
import torch
import torch.autograd as autograd
import torch.nn as nn



# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray: TypeAlias = npt.NDArray[np.floating]
IntArray: TypeAlias = npt.NDArray[np.integer]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

ScalarFieldAndValues: TypeAlias = Tuple[np.ndarray, Dict[str, float]]


def interpolate_group_geo_inr(
    *,
    group: Any,  # StructuralGroup-like (expects .name, .structural_elements, .get_interpolation_params())
    grid: Any,  # RegularGrid-like (expects .extent, .resolution, and either .grid_coordinates or .gridx/.gridy/.gridz)
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,
) -> ScalarFieldAndValues:
    """
    GeoINR interpolation for a single structural group (pure function).

    Parameters
    ----------
    group
        StructuralGroup-like object. Expected members:
        - `name: str`
        - `structural_elements: Sequence[... with .name]` in framework order (youngest -> oldest)
        - `get_interpolation_params() -> GeoINRParams-like` (expects `.beta`)
    grid
        RegularGrid-like object. Expected members:
        - `extent: tuple[float, float, float, float, float, float]`
        - `resolution: tuple[int, int, int]`
        - optionally `grid_coordinates: (N, 3) ndarray` of cell centers
        - otherwise `grid.gridx`, `grid.gridy`, `grid.gridz` 1D coordinate arrays
    group_surface_points_df
        DataFrame with columns ["X", "Y", "Z", "formation"].
    group_orientations_points_df
        DataFrame with columns ["X", "Y", "Z", "G_x", "G_y", "G_z", "formation"].

    Returns
    -------
    scalar_field : np.ndarray
        Scalar field on the grid, reshaped to `tuple(grid.resolution)` then transposed,
        preserved exactly as in the original implementation.
    scalar_values_by_element : dict[str, float]
        Mapping element_name -> scalar value (iso-values). Your original code assumes the
        returned `iso_values` are ordered oldest → youngest.

    Notes
    -----
    - Creates continuous labels in [-1, 1] mapped youngest->oldest = [+1 ... -1].
    - Uses GeoINRParams from `group.get_interpolation_params()` (expects `.beta`).
    - Does NOT mutate `group`.
    - Assumes `iso_values` returned by the model are ordered oldest→youngest.
    """
    # --- validation ---
    if group_surface_points_df is None or group_surface_points_df.empty:
        raise ValueError(f"No surface points provided for group '{group.name}'")
    if group_orientations_points_df is None or group_orientations_points_df.empty:
        raise ValueError(f"No orientations provided for group '{group.name}'")

    for col in ("X", "Y", "Z", "formation"):
        if col not in group_surface_points_df.columns:
            raise ValueError(f"Surface points for '{group.name}' missing column '{col}'")
    for col in ("X", "Y", "Z", "G_x", "G_y", "G_z", "formation"):
        if col not in group_orientations_points_df.columns:
            raise ValueError(f"Orientations for '{group.name}' missing column '{col}'")

    # --- labels: youngest -> oldest mapped to +1 ... -1 ---
    n = len(group.structural_elements)
    # youngest..oldest (group order) gets labels linearly from +1 to -1
    labels: FloatArray = np.linspace(1.0, -1.0, n)
    formation_to_label: Dict[str, float] = {
        elem.name: float(lab) for elem, lab in zip(group.structural_elements, labels)
    }

    # attach labels to surface points (as expected by your GeoINR code)
    sp = group_surface_points_df.copy()
    sp["label"] = sp["formation"].map(formation_to_label)

    # to numpy (labels first)
    interface_data: np.ndarray = sp[["label", "X", "Y", "Z"]].to_numpy()
    orientation_data: np.ndarray = group_orientations_points_df[
        ["X", "Y", "Z", "G_x", "G_y", "G_z"]
    ].to_numpy()

    # grid sampling points (N, 3)
    if hasattr(grid, "grid_coordinates") and grid.grid_coordinates is not None:
        grid_points: np.ndarray = grid.grid_coordinates
    else:
        gx, gy, gz = np.meshgrid(grid.gridx, grid.gridy, grid.gridz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # --- params ---
    params = group.get_interpolation_params()  # GeoINRParams-like (expects .beta)

    # --- run GeoINR (external function you already use) ---
    # Suppress noisy PyTorch / third-party warnings only for the duration of training.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res_inr, iso_values = stratigraphic_ConcatMLP(
            interface_data=interface_data,
            orientation_data=orientation_data,
            meshgrid_data=grid_points,
            extent=grid.extent,
            resolution=grid.resolution,
            in_dim=3,
            hidden_dim=params.hidden_dim,
            out_dim=1,
            n_hidden_layers=params.n_hidden_layers,
            activation="Softplus",
            beta=params.beta,
            concat=False,
            epochs=params.epochs,
            lr=params.lr,
            alpha=params.alpha
        )

    # `res_inr` -> (nx, ny, nz) without transpose (preserved behavior)
    scalar_field = np.asarray(res_inr).reshape(tuple(grid.resolution)).T

    # Map per-element scalar values:
    # Your previous code effectively used iso_values[0] for oldest, ... iso_values[-1] for youngest.
    # So we build the dict accordingly (oldest→youngest indexing) while your group list is youngest→oldest.
    names_old_to_young = [e.name for e in reversed(group.structural_elements)]
    if len(iso_values) != len(names_old_to_young):
        raise ValueError(
            f"GeoINR returned {len(iso_values)} iso_values, but group '{group.name}' has "
            f"{len(names_old_to_young)} elements."
        )

    scalar_values_by_element: Dict[str, float] = {
        name: float(val) for name, val in zip(names_old_to_young, iso_values)
    }

    return scalar_field, scalar_values_by_element


# -----------------------------------------------------------------------------
# Used in implicit neural representation
# -----------------------------------------------------------------------------
class ConcatMLP(nn.Module):
    """
    MLP optionally concatenating input features with hidden features at each layer.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    hidden_dim : int
        Hidden layer width.
    out_dim : int
        Output dimension (typically 1 scalar).
    n_hidden_layers : int
        Number of hidden layers (excluding input/output layers).
    activation : str
        Activation name. Supported: Softplus, ReLU, LeakyReLU, Tanh, Sigmoid, ELU, PReLU.
    beta : float
        Softplus beta parameter (only used if activation == 'Softplus').
    concat : bool
        Whether to concatenate input features with hidden representations.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_hidden_layers: int,
        activation: str,
        beta: float,
        concat: bool,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.beta = beta

        if activation == "Softplus":
            self.activation = nn.Softplus(beta=self.beta)
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        elif activation == "LeakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "Tanh":
            self.activation = nn.Tanh()
        elif activation == "Sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "ELU":
            self.activation = nn.ELU()
        elif activation == "PReLU":
            self.activation = nn.PReLU()
        else:
            # Keep print behavior unchanged; in production you might raise instead.
            print(
                "Activation function not recognized. Using Softplus, ReLU, LeakyReLU, "
                "Tanh, Sigmoid, ELU."
            )
        self.num_layers = 2 + n_hidden_layers
        self.concat = concat

        # input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if self.concat:
            # hidden layers (with concatenation)
            h_dim_concat = in_dim + hidden_dim
            for _ in range(n_hidden_layers):
                self.layers.append(nn.Linear(h_dim_concat, h_dim_concat))
                # only concatenate the input with the hidden layer features
                h_dim_concat = in_dim + h_dim_concat
            # output layer
            self.layers.append(nn.Linear(h_dim_concat, out_dim))
        else:
            # hidden layers (no concatenation)
            for _ in range(n_hidden_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # output layer
            self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, in_dim).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, out_dim).
        """
        x_target = x  # Keep the original input features for concatenation

        for i, layer in enumerate(self.layers):
            x = layer(x)

            # Only apply activation and concatenation to hidden layers (not the last layer)
            if i < len(self.layers) - 1:
                if self.concat:
                    # Concatenate the input with the hidden layer output
                    x = torch.cat((x_target, x), dim=1)
                x = self.activation(x)

        return x


def normalize(data: np.ndarray, bounds: Tuple[float, float, float, float, float, float]) -> np.ndarray:
    """
    Normalize coordinates to [-1, 1] using the provided model bounds.

    Parameters
    ----------
    data : np.ndarray
        Array of shape (N, 3) containing xyz coordinates.
    bounds : tuple[float, float, float, float, float, float]
        Extent as (xmin, xmax, ymin, ymax, zmin, zmax).

    Returns
    -------
    np.ndarray
        Normalized coordinates in [-1, 1], dtype float32.
    """
    data = data.astype(np.float32)
    mean_x = (bounds[0] + bounds[1]) / 2
    mean_y = (bounds[2] + bounds[3]) / 2
    mean_z = (bounds[4] + bounds[5]) / 2
    delta_x = bounds[1] - bounds[0]
    delta_y = bounds[3] - bounds[2]
    delta_z = bounds[5] - bounds[4]
    data[:, 0] = (data[:, 0] - mean_x) / delta_x * 2
    data[:, 1] = (data[:, 1] - mean_y) / delta_y * 2
    data[:, 2] = (data[:, 2] - mean_z) / delta_z * 2

    return data


# loss function for the interface points
def loss_intf(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """L1 loss between predicted and target interface labels."""
    criterion = nn.L1Loss()
    return criterion(y_pred, y_true)


# loss function for the orientation points
def loss_grad(
    train_x: torch.Tensor,
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    n_orien: int,
) -> torch.Tensor:
    """
    Orientation-gradient loss using autograd.

    Parameters
    ----------
    train_x : torch.Tensor
        Input tensor used for autograd (requires_grad True).
        Contains interface + orientation points (orientation points at the end).
    y_pred : torch.Tensor
        Predicted scalar value at all training points.
    y_true : torch.Tensor
        True orientation vectors for the orientation points (shape (n_orien, 3)).
    n_orien : int
        Number of orientation points (last rows in `train_x` and `y_pred`).

    Returns
    -------
    torch.Tensor
        Mean (1 - cosine similarity) over orientation points.
    """
    gradients = autograd.grad(
        outputs=y_pred,
        inputs=train_x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    grad_norm_pred = torch.norm(gradients[-n_orien:, :], p=2, dim=1)
    grad_inner_product = torch.einsum("ij, ij->i", y_true, gradients[-n_orien:, :])
    cosine = grad_inner_product / grad_norm_pred  # normal orientation
    loss_grad_val = torch.mean(1 - cosine)
    return loss_grad_val


def predict_to_mesh_stratigraphic(
    extents: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    predictions: np.ndarray,
    iso_values: np.ndarray,
) -> Tuple[pv.PolyData, pv.ImageData]:
    """
    Convert grid predictions back to a PyVista ImageData and extract isosurfaces.

    Parameters
    ----------
    extents : tuple[float, float, float, float, float, float]
        Domain extent as (xmin, xmax, ymin, ymax, zmin, zmax).
    resolution : tuple[int, int, int]
        Grid resolution as (nx, ny, nz).
    predictions : np.ndarray
        Flat predictions for all grid points.
    iso_values : np.ndarray
        Isovalues to contour.

    Returns
    -------
    contours_mean : pv.PolyData
        Contour surface(s).
    grid_mesh_final : pv.ImageData
        PyVista image grid holding the scalar field.
    """
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [
        (extents[1] - extents[0]) / (resolution[0] - 1),
        (extents[3] - extents[2]) / (resolution[1] - 1),
        (extents[5] - extents[4]) / (resolution[2] - 1),
    ]

    grid_mesh_final.point_data["scalar"] = predictions.ravel()
    contours_mean = grid_mesh_final.contour(isosurfaces=iso_values)

    return contours_mean, grid_mesh_final


def stratigraphic_ConcatMLP(
    interface_data: np.ndarray,
    orientation_data: np.ndarray,
    meshgrid_data: np.ndarray,
    extent: Tuple[float, float, float, float, float, float],
    resolution: Tuple[int, int, int],
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden_layers: int,
    activation: str = "Softplus",
    beta: float = 1,
    concat: bool = False,
    epochs: int = 2000,
    lr: float = 0.001,
    alpha: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Train a ConcatMLP to represent stratigraphic scalar field, then infer on meshgrid points.

    Parameters
    ----------
    interface_data
        Array with columns [label, x, y, z, ...]. Labels in [-1, 1].
    orientation_data
        Array with columns [x, y, z, dx, dy, dz, ...].
    meshgrid_data
        Array with columns [x, y, z, ...] for inference points.
    extent
        Model bounds as (xmin, xmax, ymin, ymax, zmin, zmax).
    resolution
        Grid resolution as (nx, ny, nz).
    in_dim, hidden_dim, out_dim, n_hidden_layers, activation, beta, concat
        Network configuration.
    epochs
        Number of training epochs.
    lr
        Learning rate.
    alpha
        Weight for orientation loss.

    Returns
    -------
    predictions : np.ndarray
        Flat predictions on `meshgrid_data` (raveled).
    iso_values : np.ndarray
        Unique interface labels (as used for isosurface extraction).
    """
    # read the input data
    train_data_x = interface_data[:, 1:].astype("float")
    train_data_y = interface_data[:, 0].astype("float")
    train_orie_x = orientation_data[:, 0:3].astype("float")
    train_orie_y = orientation_data[:, 3:6].astype("float")

    # normalize the input data
    normalized_train_data_x = normalize(train_data_x, extent)
    normalized_train_orie_x = normalize(train_orie_x, extent)
    normalized_meshgrid_data = normalize(meshgrid_data, extent)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # concatenate the interface points and orientation points for training
    train_x_dx = np.vstack((normalized_train_data_x, normalized_train_orie_x))

    # convert to torch tensor and send to GPU
    x_dx_tensor = torch.tensor(train_x_dx, dtype=torch.float32).to(device).requires_grad_(True)
    y_tensor = torch.tensor(train_data_y, dtype=torch.float32).to(device)
    dy_tensor = torch.tensor(train_orie_y, dtype=torch.float32).to(device)
    test_x_tensor = torch.tensor(normalized_meshgrid_data, dtype=torch.float32).to(device)

    # the number of interface points and orientation points for dividing the input tensor
    n_intf = normalized_train_data_x.shape[0]
    n_orie = normalized_train_orie_x.shape[0]

    # train the model
    model = ConcatMLP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        beta=beta,
        concat=concat,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Track minimum loss and corresponding parameters
    min_loss = float("inf")
    best_params = None

    t1_train = time.time()
    for epoch in range(epochs):
        y_pred = model(x_dx_tensor)

        # calculate the interface loss
        loss_i = loss_intf(y_pred[:n_intf, :].squeeze(), y_tensor)

        # calculate the orientation loss
        loss_o = loss_grad(x_dx_tensor, y_pred, dy_tensor, n_orie)

        loss = loss_i + alpha * loss_o
        if loss < min_loss:
            min_loss = loss
            best_params = model.state_dict()
            min_loss_i = loss_i
            min_loss_o = loss_o

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    t2_train = time.time()

    print(f"Training losses | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}")
    print(f"each epoch training time :  {(t2_train - t1_train) / epochs} seconds")

    # model for inference
    best_model = ConcatMLP(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        beta=beta,
        concat=concat,
    ).to(device)
    best_model.load_state_dict(best_params)

    # Predict with the model
    with torch.no_grad():
        t1_inference = time.time()
        predictions = best_model(test_x_tensor).cpu().numpy()
        t2_inference = time.time()

    # get the iso values for extracting the stratigraphic surfaces
    iso_values = np.unique(interface_data[:, 0])

    print(f"Inference time: {t2_inference - t1_inference} seconds")
    print("------Finish-------")

    return predictions.ravel(), iso_values
