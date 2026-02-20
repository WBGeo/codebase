# tests/test_interpolation_params.py

from __future__ import annotations

import math
import pytest
import numpy as np

from core.structural_modeling_components.interpolator_functions.interpolator_parameters import (  # type: ignore
    InterpolationContext,
    InterpolationMethod,
    OKParams,
    RBFParams,
    UKParams,
    UCKParams,
    GeoINRParams,
    FDIParams,
    PLIParams,
    default_ok_params,
    default_rbf_params,
    default_uk_params,
    default_uck_params,
    default_geo_inr_params,
    default_fdi_params,
    default_pli_params,
)

# -----------------------------------------------------------------------------
# Enums and model defaults
# -----------------------------------------------------------------------------

def test_interpolation_method_enum_values_are_strings():
    # Guard against accidental renames that break config files / serialized values
    assert InterpolationMethod.ORDINARY_KRIGING.value == "Ordinary Kriging"
    assert InterpolationMethod.RADIAL_BASIS_FUNCTION.value == "Radial Basis Function"
    assert InterpolationMethod.UNIVERSAL_KRIGING.value == "Universal Kriging"
    # LoopStructural backends exist in the enum
    assert InterpolationMethod.FINITE_DIFFERENCES.value == "Finite Differences"
    assert InterpolationMethod.PIECEWISE_LINEAR.value == "Piecewise Linear"


def test_pydantic_param_models_have_expected_defaults():
    ok = OKParams()
    assert ok.variogram_model == "gaussian"
    assert ok.range == 500.0
    assert ok.sill == 1.0
    assert ok.nugget == 0.0
    assert ok.neighbors is None

    rbf = RBFParams()
    assert rbf.kernel == "thin_plate_spline"
    assert rbf.smoothing == 0
    assert rbf.epsilon is None
    assert rbf.neighbors is None

    uk = UKParams()
    # Inherits OK defaults + adds drift_terms
    assert uk.variogram_model == "gaussian"
    assert uk.range == 500.0
    assert uk.drift_terms == "regional_linear"

    fdi = FDIParams()
    assert fdi.nelements == 10_000
    assert fdi.solver == "cg"
    assert fdi.damp is True
    assert fdi.tol is None

    pli = PLIParams()
    assert pli.nelements == 5_000
    assert pli.solver == "cg"
    assert pli.damp is True
    assert pli.tol is None


# -----------------------------------------------------------------------------
# Default-parameter heuristics
# -----------------------------------------------------------------------------

def test_default_ok_params_neighbors_none_when_npts_lt_20():
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=19, mean_nn_distance=2.0)
    ok = default_ok_params(ctx)
    assert ok.neighbors is None


def test_default_ok_params_neighbors_window_when_npts_ge_20():
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=200, mean_nn_distance=2.0)
    ok = default_ok_params(ctx)
    # max(30, npts//10)=max(30,20)=30; min(200,30)=30
    assert ok.neighbors == 30


def test_default_ok_params_range_is_clipped_to_bounds():
    # Range = clip(20*nn, 0.1*max_scale, 0.8*max_scale)
    ctx = InterpolationContext(data_scale=(100.0, 50.0, 25.0), n_points=10, mean_nn_distance=0.1)
    ok = default_ok_params(ctx)

    max_scale = 100.0
    lower = 0.1 * max_scale  # 10
    # 20*0.1 = 2 -> clipped to 10
    assert math.isclose(ok.range, lower)


def test_default_ok_params_anisotropy_scaling_is_clipped():
    # anisotropy_scaling_y = clip(sy/sx, 0.05, 1.0)
    # anisotropy_scaling_z = clip(sz/sx, 0.05, 1.0)
    ctx = InterpolationContext(data_scale=(100.0, 1.0, 1e-6), n_points=10, mean_nn_distance=1.0)
    ok = default_ok_params(ctx)
    assert math.isclose(ok.anisotropy_scaling_y, 0.05)  # 0.01 -> clipped up
    assert math.isclose(ok.anisotropy_scaling_z, 0.05)  # ~0 -> clipped up


def test_default_rbf_params_neighbors_only_for_very_large_npts():
    ctx_small = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1000, mean_nn_distance=2.0)
    rbf_small = default_rbf_params(ctx_small)
    assert rbf_small.neighbors is None

    ctx_big = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=6000, mean_nn_distance=2.0)
    rbf_big = default_rbf_params(ctx_big)
    assert rbf_big.neighbors is not None
    assert 0 < rbf_big.neighbors <= 500


def test_default_rbf_params_kernel_and_smoothing_defaults():
    ctx = InterpolationContext(data_scale=(100.0, 50.0, 25.0), n_points=10, mean_nn_distance=2.0)
    rbf = default_rbf_params(ctx)
    assert rbf.kernel == "thin_plate_spline"
    assert math.isclose(rbf.smoothing, 0.05)


def test_default_uk_params_neighbors_rule_matches_ok():
    ctx1 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=19, mean_nn_distance=2.0)
    uk1 = default_uk_params(ctx1)
    assert uk1.neighbors is None

    ctx2 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=250, mean_nn_distance=2.0)
    uk2 = default_uk_params(ctx2)
    # max(30,25)=30; min(200,30)=30
    assert uk2.neighbors == 30


def test_default_uk_params_range_is_clipped_and_typically_longer_than_ok():
    # OK uses 20*nn, UK uses 30*nn with slightly different bounds.
    ctx = InterpolationContext(data_scale=(100.0, 80.0, 60.0), n_points=100, mean_nn_distance=2.0)

    ok = default_ok_params(ctx)
    uk = default_uk_params(ctx)

    # In this regime neither should hit hard clipping, so UK should be >= OK
    assert uk.range >= ok.range

    # UK clip bounds: [0.15*max, 0.9*max]
    max_scale = 100.0
    assert (0.15 * max_scale) <= uk.range <= (0.9 * max_scale)


def test_default_uk_params_anisotropy_uses_safe_division_when_sx_is_zeroish():
    # sx_safe = max(sx, 1e-12), so no division-by-zero
    ctx = InterpolationContext(data_scale=(0.0, 10.0, 10.0), n_points=100, mean_nn_distance=1.0)
    uk = default_uk_params(ctx)
    assert np.isfinite(uk.anisotropy_scaling_y)
    assert np.isfinite(uk.anisotropy_scaling_z)
    # clipped into [0.05, 20.0]
    assert 0.05 <= uk.anisotropy_scaling_y <= 20.0
    assert 0.05 <= uk.anisotropy_scaling_z <= 20.0


def test_default_uk_params_sets_drift_terms_regional_linear():
    ctx = InterpolationContext(data_scale=(100.0, 80.0, 60.0), n_points=100, mean_nn_distance=2.0)
    uk = default_uk_params(ctx)
    assert uk.drift_terms == "regional_linear"


# -----------------------------------------------------------------------------
# LoopStructural defaults (split FDI / PLI)
# -----------------------------------------------------------------------------

def test_default_fdi_params_nelements_within_bounds():
    for n in [1, 10, 1_000, 10_000, 10_000_000]:
        ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=n, mean_nn_distance=2.0)
        p = default_fdi_params(ctx)
        assert 5_000 <= p.nelements <= 200_000, f"n={n}: nelements={p.nelements} out of [5k, 200k]"

    # Check fixed fields
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1_000, mean_nn_distance=2.0)
    p = default_fdi_params(ctx)
    assert p.solver == "cg"
    assert p.damp is True
    assert p.tol is None


def test_default_fdi_params_exact_at_1k_points():
    """At n=1000: scale=(1000/1000)^0.5=1.0, factor=max(0.7,min(1.0,8.0))=1.0, nelements=int(10000*1.0)=10000."""
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1_000, mean_nn_distance=2.0)
    p = default_fdi_params(ctx)
    assert p.nelements == 10_000


def test_default_fdi_params_grows_with_npoints():
    """nelements is non-decreasing as n_points increases."""
    base_ctx = dict(data_scale=(100.0, 100.0, 100.0), mean_nn_distance=2.0)
    ns = [100, 1_000, 10_000, 1_000_000]
    params = [default_fdi_params(InterpolationContext(**base_ctx, n_points=n)) for n in ns]
    for smaller, larger in zip(params, params[1:]):
        assert smaller.nelements <= larger.nelements


def test_default_pli_params_nelements_within_bounds():
    # Bounds are [2k, 50k] after clamping
    for n in [10, 100, 1_000, 10_000, 10_000_000]:
        ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=n, mean_nn_distance=2.0)
        p = default_pli_params(ctx)
        assert 2_000 <= p.nelements <= 50_000, f"n={n}: nelements={p.nelements} out of [2k, 50k]"

    # Check fixed fields are always set correctly
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1_000, mean_nn_distance=2.0)
    p = default_pli_params(ctx)
    assert p.solver == "cg"
    assert p.damp is True
    assert p.tol is None


def test_default_pli_params_floor_clamping():
    """Tiny n_points: scale_n << 1, factor clamped to 0.8 → raw nelements < 2000 → floored to 2000."""
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10, mean_nn_distance=2.0)
    p = default_pli_params(ctx)
    assert p.nelements == 2_000


def test_default_pli_params_grows_with_npoints():
    """nelements is non-decreasing as n_points increases (same scale / nn_distance)."""
    base_ctx = dict(data_scale=(100.0, 100.0, 100.0), mean_nn_distance=2.0)
    ns = [100, 1_000, 10_000, 100_000]
    params = [default_pli_params(InterpolationContext(**base_ctx, n_points=n)) for n in ns]
    for smaller, larger in zip(params, params[1:]):
        assert smaller.nelements <= larger.nelements


def test_default_pli_params_density_adjustment_increases_nelements():
    """Denser sampling (smaller nn_dist relative to extent) produces >= nelements."""
    ctx_sparse = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1_000, mean_nn_distance=10.0)
    ctx_dense  = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1_000, mean_nn_distance=0.01)
    p_sparse = default_pli_params(ctx_sparse)
    p_dense  = default_pli_params(ctx_dense)
    assert p_dense.nelements >= p_sparse.nelements


# -----------------------------------------------------------------------------
# UCK + GeoINR defaults (smoke tests)
# -----------------------------------------------------------------------------

def test_default_uck_params_returns_instance():
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=100, mean_nn_distance=2.0)
    uck = default_uck_params(ctx)
    assert isinstance(uck, UCKParams)


def test_default_geo_inr_params_returns_instance_and_has_reasonable_ranges():
    ctx = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1000, mean_nn_distance=2.0)
    p = default_geo_inr_params(ctx)
    assert isinstance(p, GeoINRParams)

    # conservative sanity checks (don’t over-specify)
    assert p.hidden_dim >= 4
    assert p.n_hidden_layers >= 0
    assert p.epochs >= 1
    assert p.lr > 0.0
    assert p.alpha >= 0.0
    assert p.beta >= 0.01


def test_default_geo_inr_params_capacity_breakpoints():
    """Verify hidden_dim / n_hidden_layers at each capacity breakpoint."""
    # n < 500: hidden_dim=32, n_hidden_layers=2
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=100, mean_nn_distance=2.0))
    assert p.hidden_dim == 32
    assert p.n_hidden_layers == 2

    # 500 <= n < 3000: hidden_dim=32, n_hidden_layers=1
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=600, mean_nn_distance=2.0))
    assert p.hidden_dim == 32
    assert p.n_hidden_layers == 1

    # n >= 3000: hidden_dim=64, n_hidden_layers=2
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=5_000, mean_nn_distance=2.0))
    assert p.hidden_dim == 64
    assert p.n_hidden_layers == 2


def test_default_geo_inr_params_epoch_lr_breakpoints():
    """Verify epochs / lr at each training breakpoint."""
    # n < 800: epochs=6000, lr=0.01
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=100, mean_nn_distance=2.0))
    assert p.epochs == 6_000
    assert math.isclose(p.lr, 0.01)

    # 800 <= n < 5000: epochs=4000, lr=0.005
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=2_000, mean_nn_distance=2.0))
    assert p.epochs == 4_000
    assert math.isclose(p.lr, 0.005)

    # n >= 5000: epochs=2500, lr=0.003
    p = default_geo_inr_params(InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10_000, mean_nn_distance=2.0))
    assert p.epochs == 2_500
    assert math.isclose(p.lr, 0.003)
