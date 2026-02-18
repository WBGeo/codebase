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
    assert pli.nelements == 20_000
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

def test_default_fdi_params_nelements_scaling_and_bounds():
    # n=1000 -> around base (10k)
    ctx1 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1000, mean_nn_distance=2.0)
    p1 = default_fdi_params(ctx1)
    assert 5_000 <= p1.nelements <= 200_000
    assert p1.solver == "cg"
    assert p1.damp is True
    assert p1.tol is None

    # n very small -> should still be clamped >= 5000
    ctx2 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10, mean_nn_distance=2.0)
    p2 = default_fdi_params(ctx2)
    assert p2.nelements >= 5_000

    # n huge -> should clamp to <= 200k
    ctx3 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10_000_000, mean_nn_distance=2.0)
    p3 = default_fdi_params(ctx3)
    assert p3.nelements <= 200_000


def test_default_pli_params_nelements_scaling_and_bounds():
    # moderate
    ctx1 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=1000, mean_nn_distance=2.0)
    p1 = default_pli_params(ctx1)
    assert 10_000 <= p1.nelements <= 400_000
    assert p1.solver == "cg"
    assert p1.damp is True
    assert p1.tol is None

    # very small -> should still clamp >= 10k
    ctx2 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10, mean_nn_distance=2.0)
    p2 = default_pli_params(ctx2)
    assert p2.nelements >= 10_000

    # huge -> should clamp <= 400k
    ctx3 = InterpolationContext(data_scale=(100.0, 100.0, 100.0), n_points=10_000_000, mean_nn_distance=2.0)
    p3 = default_pli_params(ctx3)
    assert p3.nelements <= 400_000

    # sanity: PLI defaults should generally be >= FDI defaults for the same context
    fdi = default_fdi_params(ctx1)
    assert p1.nelements >= fdi.nelements


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
