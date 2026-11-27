"""GeoINR training: reproducibility (seed), best-epoch weights, coordinate precision."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import core.structural_modeling_components.interpolator_functions.geoinr as geoinr

_EXTENT = (0.0, 4.0, 0.0, 3.0, 0.0, 2.0)
_RESOLUTION = (4, 3, 2)


def _data():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0.2, 2.8, (10, 2))
    interface = np.vstack([
        np.column_stack([np.full(10, 1.0), xy, np.full(10, 1.4)]),
        np.column_stack([np.full(10, -1.0), xy, np.full(10, 0.6)]),
    ])
    orientations = np.array([[2.0, 1.5, 1.0, 0.0, 0.0, 1.0]])
    c = [np.linspace(lo + 0.5, hi - 0.5, n) for (lo, hi), n in zip(np.reshape(_EXTENT, (3, 2)), _RESOLUTION)]
    gx, gy, gz = np.meshgrid(*c, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    return interface, orientations, grid


def _train(seed, epochs=30):
    interface, orientations, grid = _data()
    return geoinr.stratigraphic_ConcatMLP(
        interface_data=interface, orientation_data=orientations, meshgrid_data=grid,
        extent=_EXTENT, resolution=_RESOLUTION, in_dim=3, hidden_dim=8, out_dim=1,
        n_hidden_layers=1, epochs=epochs, lr=0.01, alpha=0.1, n_eikonal_samples=10, seed=seed,
    )[0]


def test_same_seed_gives_same_result_and_leaves_global_random_state_alone():
    torch.manual_seed(123)
    np.random.seed(123)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()[1].copy()

    a, b, c = _train(seed=0), _train(seed=0), _train(seed=1)
    assert np.array_equal(a, b)
    assert not np.allclose(a, c)

    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state)


def test_returns_weights_of_the_lowest_loss_epoch(monkeypatch):
    """
    The loss is made to be lowest in epoch 0; the prediction must come from the weights
    of epoch 0, not from the weights after the last epoch.
    """
    calls = {"n": 0}
    real_loss_intf = geoinr.loss_intf

    def loss_lowest_first(y_pred, y_true):
        calls["n"] += 1
        return real_loss_intf(y_pred, y_true) * (0.0 if calls["n"] == 1 else 1.0) + (calls["n"] > 1) * 10.0

    monkeypatch.setattr(geoinr, "loss_intf", loss_lowest_first)
    after_epoch_0 = _train(seed=0, epochs=1)
    calls["n"] = 0
    after_many = _train(seed=0, epochs=20)
    assert np.allclose(after_many, after_epoch_0)


def test_formation_without_surface_points_still_gets_its_value():
    """E.g. in a fault domain: labels are fixed per formation, not taken from the data."""
    from types import SimpleNamespace

    from core.structural_modeling_components.interpolator_functions.interpolator_parameters import GeoINRParams

    interface, orientations, grid_points = _data()
    names = ["top", "middle", "base"]  # youngest -> oldest
    group = SimpleNamespace(
        name="G",
        structural_elements=[SimpleNamespace(name=n) for n in names],
        get_interpolation_params=lambda: GeoINRParams(epochs=5, hidden_dim=8),
    )
    surface_points = pd.DataFrame({  # no points of "middle"
        "X": interface[:, 1], "Y": interface[:, 2], "Z": interface[:, 3],
        "formation": np.where(interface[:, 0] > 0, "top", "base"),
    })
    orientation_df = pd.DataFrame(orientations, columns=["X", "Y", "Z", "G_x", "G_y", "G_z"]).assign(formation="top")
    grid = SimpleNamespace(extent=_EXTENT, resolution=_RESOLUTION, grid_coordinates=grid_points)

    field, values = geoinr.interpolate_group_geo_inr(
        group=group, grid=grid, group_surface_points_df=surface_points, group_orientations_points_df=orientation_df,
    )
    assert values == {"top": 1.0, "middle": 0.0, "base": -1.0}
    assert field.shape == tuple(_RESOLUTION)[::-1]


def test_single_formation_orientations_define_up():
    """One formation: the orientation (pointing up) makes the field increase upwards."""
    from types import SimpleNamespace

    from core.structural_modeling_components.interpolator_functions.interpolator_parameters import GeoINRParams

    interface, orientations, grid_points = _data()
    top = interface[interface[:, 0] > 0]
    group = SimpleNamespace(
        name="G", structural_elements=[SimpleNamespace(name="only")],
        get_interpolation_params=lambda: GeoINRParams(epochs=300),
    )
    surface_points = pd.DataFrame({"X": top[:, 1], "Y": top[:, 2], "Z": top[:, 3], "formation": "only"})
    orientation_df = pd.DataFrame(orientations, columns=["X", "Y", "Z", "G_x", "G_y", "G_z"]).assign(formation="only")
    grid = SimpleNamespace(extent=_EXTENT, resolution=_RESOLUTION, grid_coordinates=grid_points)

    field, values = geoinr.interpolate_group_geo_inr(
        group=group, grid=grid, group_surface_points_df=surface_points, group_orientations_points_df=orientation_df,
    )
    assert values == {"only": 1.0}
    assert field[1].mean() > field[0].mean()  # ZYX: upper layer above lower layer


def test_normalize_keeps_precision_of_real_world_coordinates():
    offset = np.array([523_000.0, 5_810_000.0, 1_200.0])
    extent = (offset[0], offset[0] + 4.0, offset[1], offset[1] + 3.0, offset[2], offset[2] + 2.0)
    points = offset + np.array([[0.01, 0.02, 0.03], [3.99, 2.98, 1.97]])
    expected = geoinr.normalize(points - offset, _EXTENT)
    assert np.allclose(geoinr.normalize(points, extent), expected, atol=1e-6)
