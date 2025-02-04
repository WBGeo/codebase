import numpy as np
from core.object_components import InputData, GeomodelResults
from core.utility.surface_mesh_extraction import marching_cubes
from skimage import measure

import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pyvista as pv
import pyvistaqt as pvqt
import pandas as pd
import time
import torch.autograd as autograd


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


def geo_inr_interpolator(input_data: InputData, beta: int = 10):
    """
    Compute a model based on input data using geoINR interpolation

    Args:
        input_data (InputData): The input data for the geological model.
        beta (int): Beta parameter for the Softplus activation function.

    Returns:
        resultsGeomodelResults: The results of the geological model.

    """
    # Based on input data create regular grid - this can be outsourced to a separate function
    dx = (input_data.extent[1] - input_data.extent[0]) / input_data.resolution[0]
    dy = (input_data.extent[3] - input_data.extent[2]) / input_data.resolution[1]
    dz = (input_data.extent[5] - input_data.extent[4]) / input_data.resolution[2]
    spacing = (dx, dy, dz)
    gridx = np.linspace(input_data.extent[0] + dx / 2, input_data.extent[1] - dx / 2, input_data.resolution[0])
    gridy = np.linspace(input_data.extent[2] + dy / 2, input_data.extent[3] - dy / 2, input_data.resolution[1])
    gridz = np.linspace(input_data.extent[4] + dz / 2, input_data.extent[5] - dz, input_data.resolution[2])

    # gempy way to get coordinates, for some reason this does not blow memory
    coords = gridx, gridy, gridz
    g = np.meshgrid(*coords, indexing="ij")
    grid = np.vstack(tuple(map(np.ravel, g))).T.astype("float64")

    # Create mapping for replacing element names with ints
    unique_elements = sorted(set(element for elements in input_data.mapping_object.values() for element in elements))
    replacements = {element: i + 1 for i, element in enumerate(unique_elements)}

    # Map formations to values between -1 and 1
    surface_points = input_data.surface_points.copy()
    replacements_inr = {key: value for key, value in zip(replacements.keys(), np.linspace(-1, 1, len(replacements)))}
    surface_points['label'] = surface_points['formation'].replace(replacements_inr)

    # Separate data based on structural groups
    results = []
    masks = []
    counter = 0

    # Reverse the order of the keys
    reversed_dict = {key: input_data.mapping_object[key] for key in reversed(input_data.mapping_object)}

    # for key, value in input_data.mapping_object.items():
    for key, value in reversed_dict.items():
        # Create a new dataframe with the structural group
        structural_group_df = surface_points[surface_points['formation'].isin(list(input_data.mapping_object[key]))]
        orientations_group_df = input_data.orientations[input_data.orientations['formation'].isin(list(input_data.mapping_object[key]))]

        structural_group_df.loc[
            structural_group_df['formation'].isin(list(input_data.mapping_object[key])), 'formation'] = \
            structural_group_df['formation'].replace(replacements)

        surface_points_group = structural_group_df[['label', 'X', 'Y', 'Z']].values  # labels in the first column
        orientations_group = orientations_group_df[['X', 'Y', 'Z', 'G_x', 'G_y', 'G_z']].values

        # perform geoINR per structural group
        res_inr, iso_values = stratigraphic_ConcatMLP(interface_data=surface_points_group,
                                                      orientation_data=orientations_group,
                                                      meshgrid_data=grid,
                                                      extent=input_data.extent,  # domain boundary
                                                      resolution=input_data.resolution,
                                                      in_dim=3,  # input dimension of neural network
                                                      hidden_dim=32,
                                                      out_dim=1,
                                                      n_hidden_layers=1,  # number of hidden layers
                                                      activation='Softplus',
                                                      beta=beta,
                                                      concat=False,
                                                      epochs=1000,
                                                      lr=0.01)  # learning rate

        res_inr = res_inr.reshape(input_data.resolution)

        import matplotlib.pyplot as plt
        # plot_block = res_inr
        # image = plot_block[:, int(np.rint(input_data.resolution[1] / 2)), :].T
        # plt.imshow(image, origin='lower', cmap='viridis')
        # plt.colorbar()
        # plt.show()

        # Replace values with integers based on iso values
        # TODO: check if this works for many groups/more elements
        new_res_inr = np.zeros(res_inr.shape)
        # iso_values = np.sort(iso_values)  # Ensure iso_values is sorted
        for i in range(len(iso_values)-1):
            new_res_inr[(res_inr >= iso_values[i]) & (res_inr < iso_values[i + 1])] = counter + i + 1

        new_res_inr[res_inr >= iso_values[-1]] = len(iso_values) + counter

        counter = counter + len(value)

        # plot_block = new_res_inr
        # image = plot_block[:, int(np.rint(input_data.resolution[1] / 2)), :].T
        # plt.imshow(image, origin='lower', cmap='viridis')
        # plt.colorbar()
        # plt.show()

        # Save results
        results.append(new_res_inr.astype(int))

        # Create mask for values below the lowest integer value for stacking
        mask = new_res_inr.astype(int) > 0
        masks.append(mask)

    # Stack result based on stack
    combined_result = np.zeros(results[0].shape)

    # Iterate over the results and masks arrays in reverse order
    for i in reversed(range(len(results) - 1, -1, -1)):
        combined_result[masks[i]] = results[i][masks[i]]

    # Reverse everything to match gempy, probably have to rewrite everything at some point
    max_val = int(np.max(combined_result))
    mapping = {i: max_val - i for i in range(max_val + 1)}

    # Apply the mapping to the array
    combined_result = np.vectorize(mapping.get)(combined_result)

    # plot_block = combined_result
    # image = plot_block[:, int(np.rint(input_data.resolution[1] / 2)), :].T
    # plt.imshow(image, origin='lower', cmap='viridis')
    # plt.colorbar()
    # plt.show()

    # Extract the surface meshes using marching cubes, does not consider faults as not possible atm
    mc_vertices, mc_edges = marching_cubes(combined_result, unique_elements, spacing, input_data.extent)

    # Create a GeomodelResults instance
    results_instance = GeomodelResults(name=input_data.name,
                                       lith_block=combined_result.flatten(),
                                       surface_meshes_vertices=mc_vertices,
                                       surface_meshes_edges=mc_edges,
                                       grid=grid,
                                       extent=input_data.extent,
                                       resolution=input_data.resolution,
                                       mapping_object=input_data.mapping_object)

    return results_instance
