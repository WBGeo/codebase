
from core.structural_objects.objects import StructuralGroup

import warnings

import torch
import torch.nn as nn
import numpy as np
import pyvista as pv
import pandas as pd
import time
import torch.autograd as autograd

warnings.filterwarnings("ignore")

from typing import Dict, Tuple, Optional


def interpolate_group_geo_inr(
    *,
    group,                        # StructuralGroup
    grid,                         # RegularGrid (has extent, resolution, grid_coordinates OR (grid.gridx, grid.gridy, grid.gridz))
    group_surface_points_df: pd.DataFrame,
    group_orientations_points_df: Optional[pd.DataFrame] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    GeoINR interpolation for a single structural group (pure function).

    Returns:
        scalar_field : np.ndarray with shape == tuple(grid.resolution)  (nx, ny, nz)
        scalar_values_by_element : Dict[str, float] mapping element_name -> scalar value

    Notes:
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
    labels = np.linspace(1.0, -1.0, n)
    formation_to_label = {
        elem.name: lab for elem, lab in zip(group.structural_elements, labels)
    }

    # attach labels to surface points (as expected by your GeoINR code)
    sp = group_surface_points_df.copy()
    sp["label"] = sp["formation"].map(formation_to_label)

    # to numpy (labels first)
    interface_data = sp[["label", "X", "Y", "Z"]].to_numpy()
    orientation_data = group_orientations_points_df[["X", "Y", "Z", "G_x", "G_y", "G_z"]].to_numpy()

    # grid sampling points (N, 3)
    if hasattr(grid, "grid_coordinates") and grid.grid_coordinates is not None:
        grid_points = grid.grid_coordinates
    else:
        gx, gy, gz = np.meshgrid(grid.gridx, grid.gridy, grid.gridz, indexing="ij")
        grid_points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    # --- params ---
    params = group.get_interpolation_params()  # GeoINRParams (expects .beta)

    # --- run GeoINR (external function you already use) ---
    res_inr, iso_values = stratigraphic_ConcatMLP(
        interface_data=interface_data,
        orientation_data=orientation_data,
        meshgrid_data=grid_points,
        extent=grid.extent,
        resolution=grid.resolution,
        in_dim=3,
        hidden_dim=32,
        out_dim=1,
        n_hidden_layers=1,
        activation='Softplus',
        beta=params.beta,
        concat=False,
        epochs=5000,
        lr=0.01,
    )

    # `res_inr` -> (nx, ny, nz) without transpose
    scalar_field = np.asarray(res_inr).reshape(tuple(grid.resolution))

    # Map per-element scalar values:
    # Your previous code effectively used iso_values[0] for oldest, ... iso_values[-1] for youngest.
    # So we build the dict accordingly (oldest→youngest indexing) while your group list is youngest→oldest.
    names_old_to_young = [e.name for e in reversed(group.structural_elements)]
    if len(iso_values) != len(names_old_to_young):
        raise ValueError(
            f"GeoINR returned {len(iso_values)} iso_values, but group '{group.name}' has {len(names_old_to_young)} elements."
        )

    scalar_values_by_element = {
        name: float(val) for name, val in zip(names_old_to_young, iso_values)
    }

    return scalar_field, scalar_values_by_element



# used in implicit neural representation
class ConcatMLP(nn.Module):
    """
    concatenate the input features with the hidden layer features as an enhanced feature
    the neural network structure is flexible, where
    in_dim: the input dimension, the coodinates plus fault features
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    """

    def __init__(self,
                 in_dim,
                 hidden_dim,
                 out_dim,
                 n_hidden_layers,
                 activation,
                 beta,
                 concat):
        super(ConcatMLP, self).__init__()
        self.layers = nn.ModuleList()
        self.beta = beta
        if activation == 'Softplus':
            self.activation = nn.Softplus(beta=self.beta)
        elif activation == 'ReLU':
            self.activation = nn.ReLU()
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU()
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'ELU':
            self.activation = nn.ELU()
        elif activation == 'PReLU':
            self.activation = nn.PReLU()
        else:
            print('Activation function not recognized. Using Softplus, ReLU, LeakyReLU, Tanh, Sigmoid, ELU.')
        self.num_layers = 2 + n_hidden_layers
        self.concat = concat

        # input layer
        self.layers.append(nn.Linear(in_dim, hidden_dim))

        if self.concat:
            # hidden layers
            h_dim_concat = in_dim + hidden_dim
            for i in range(n_hidden_layers):
                self.layers.append(nn.Linear(h_dim_concat, h_dim_concat))
                # h_dim_concat *= 2  # concatenate all the former layer's features
                h_dim_concat = in_dim + h_dim_concat  # only concatenate the input with the hidden layer features
            # output layer
            self.layers.append(nn.Linear(h_dim_concat, out_dim))

        else:
            # hidden layers
            for i in range(n_hidden_layers):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            # output layer
            self.layers.append(nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
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


def normalize(data, bounds):
    """
    Normalize the data to [-1, 1]
    data: the data to be normalized
    bounds: the extent of the model domain
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
def loss_intf(y_pred, y_true):
    #criterion = nn.MSELoss()
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss(reduction='sum')

    return criterion(y_pred, y_true)


# loss function for the orientation points
def loss_grad(train_x, y_pred, y_true, n_orien):
    """
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[
        0]
    grad_norm_pred = torch.norm(gradients[-n_orien:, :], p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', y_true, gradients[-n_orien:, :])
    cosine = grad_inner_product / grad_norm_pred  # orie_tensor using normal orientation
    #loss_grad = torch.sum(1 - cosine)
    loss_grad = torch.mean(1 - cosine)

    return loss_grad


# loss function for the interface points
def loss_intf(y_pred, y_true):
    #criterion = nn.MSELoss()
    #criterion = nn.MSELoss(reduction='sum')
    criterion = nn.L1Loss()
    #criterion = nn.SmoothL1Loss(reduction='sum')

    return criterion(y_pred, y_true)


# loss function for the orientation points
def loss_grad(train_x, y_pred, y_true, n_orien):
    """
    train_x: the input data, includes the interface points and orientation points, the orientaion points are at the end of the input data
             train_x = [interface points, orientation points]
    y_pred: the predicted orientation
    y_true: the true orientation
    n_orien: the number of orientation points
    """
    gradients = autograd.grad(outputs=y_pred, inputs=train_x, grad_outputs=torch.ones_like(y_pred), create_graph=True)[
        0]
    grad_norm_pred = torch.norm(gradients[-n_orien:, :], p=2, dim=1)
    grad_inner_product = torch.einsum('ij, ij->i', y_true, gradients[-n_orien:, :])
    cosine = grad_inner_product / grad_norm_pred  # orie_tensor using normal orientation
    #loss_grad = torch.sum(1 - cosine)
    loss_grad = torch.mean(1 - cosine)

    return loss_grad


def predict_to_mesh_stratigraphic(extents, resolution, predictions, iso_values):
    """
    the prediction results of gridmesh points are assigned back to the gridmesh points under the same resolution
    multiple isovalues are used to generate the stratigraphic surfaces according to the 'iso_values'
    extents: the extent of the model domain
    resolution: the resolution of the model
    predictions: the prediction results of the gridmesh points
    """
    grid_mesh_final = pv.ImageData()
    grid_mesh_final.dimensions = resolution
    grid_mesh_final.origin = [extents[0], extents[2], extents[4]]
    grid_mesh_final.spacing = [(extents[1] - extents[0]) / (resolution[0] - 1),
                               (extents[3] - extents[2]) / (resolution[1] - 1),
                               (extents[5] - extents[4]) / (resolution[2] - 1)]

    grid_mesh_final.point_data['scalar'] = predictions.ravel()
    contours_mean = grid_mesh_final.contour(isosurfaces=iso_values)

    return contours_mean, grid_mesh_final


def stratigraphic_ConcatMLP(interface_data, orientation_data, meshgrid_data, extent, resolution, in_dim, hidden_dim,
                            out_dim,
                            n_hidden_layers, activation='Softplus', beta=1, concat=False, epochs=2000, lr=0.001,
                            alpha=0.1):
    """
    Notes: 1. this function is used to model the stratigraphic surfaces, fault feature encoding for this purpose
           2. use 'autograd' to calculate the orientation gradient
    interface_data: the data comes from the feature encoding results, the list is [label, x, y, z, fault1, fault2, ...],
                    label value in range [-1, 1], fault1, fault2 are the fault feature encoding results
    orientation_data: the data comes from the feature encoding results, the list is [x, y, z, dx, dy, dz, fault1, fault2, ...],
    meshgrid_data: the meshgrid points for predicting the domain, the format is [x, y, z, fault1, fault2, ...]
    extent: the extent of the model
    resolution: the resolution of the model
    in_dim: the input dimension of the neural network
    hidden_dim: the hidden layer dimension
    out_dim: the output dimension, a scalar value
    n_hidden_layers: the number of hidden layers
    activation: the activation function, default is 'Softplus'
    beta: the beta parameter in the Softplus activation function, effective when the activation function is Softplus
    concat: whether to concatenate the input features with the hidden layer features
    epochs: the number of epochs for training the model
    lr: the learning rate for training the model
    delta_orie: the delta value for calculating the orientation gradient, set as 1 means the gradient is calculated by the difference of 1 cell
    alpha: the weight of the orientation loss, default is 0.1
    """
    # read the data
    train_data_x = interface_data[:, 1:].astype("float")
    train_data_y = interface_data[:, 0].astype("float")
    train_orie_x = orientation_data[:, 0:3].astype("float")
    train_orie_y = orientation_data[:, 3:6].astype("float")
    # normalize the data
    normalized_train_data_x = normalize(train_data_x, extent)
    normalized_train_orie_x = normalize(train_orie_x, extent)
    normalized_meshgrid_data = normalize(meshgrid_data, extent)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # concatenate the interface points and orientation points for training
    train_x_dx = np.vstack((normalized_train_data_x, normalized_train_orie_x))
    # convert to torch tensor and send to GPU
    x_dx_tensor = torch.tensor(train_x_dx, dtype=torch.float32).to(device).requires_grad_(True)
    y_tensor = torch.tensor(train_data_y, dtype=torch.float32).to(device)
    dy_tensor = torch.tensor(train_orie_y, dtype=torch.float32).to(device)
    test_x_tensor = torch.tensor(normalized_meshgrid_data, dtype=torch.float32).to(device)
    # the number of interface points and orientation points for dividing the input tensor to send to loss functions
    n_intf = normalized_train_data_x.shape[0]
    n_orie = normalized_train_orie_x.shape[0]
    # train the model
    model = ConcatMLP(in_dim=in_dim,
                      hidden_dim=hidden_dim,
                      out_dim=out_dim,
                      n_hidden_layers=n_hidden_layers,
                      activation=activation,
                      beta=beta,
                      concat=concat,
                      ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)  # Adam
    # Initialize variables to track minimum loss and corresponding parameters
    min_loss = float('inf')
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
            # Save the current parameters of the model
            best_params = model.state_dict()
            min_loss_i = loss_i
            min_loss_o = loss_o
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    t2_train = time.time()

    print(f'Training losses | Loss_i: {min_loss_i.item()}, Loss_o: {min_loss_o.item()}')
    print(f'each epoch training time :  {(t2_train - t1_train) / epochs} seconds')
    # model for inference
    best_model = ConcatMLP(in_dim=in_dim,
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
    # convert to pyvista mesh
    # get the iso values for extracting the stratigraphic surfaces
    iso_values = np.unique(interface_data[:, 0])
    # stratigraphic_mesh, grid_mesh_final = predict_to_mesh_stratigraphic(extent, resolution, predictions, iso_values)
    print(f'Inference time: {t2_inference - t1_inference} seconds')
    print('------Finish-------')

    return predictions.ravel(), iso_values  # stratigraphic_mesh, grid_mesh_final
