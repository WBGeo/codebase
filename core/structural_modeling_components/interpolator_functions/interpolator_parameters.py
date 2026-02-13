from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from pydantic import BaseModel, Field, PrivateAttr


# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]

MeshType3 = str  # expected: "masked" | "unmasked" | "combined"
MeshType2 = str  # expected: "masked" | "unmasked"
MeshDict = Dict[str, npt.NDArray[np.generic]]


class InterpolationMethod(str, Enum):
    """Supported interpolation backends for structural scalar fields."""

    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    LOOP_STRUCTURAL = "Loop Structural"
    UNIVERSAL_KRIGING = "Universal Kriging"


class InterpolationContext(BaseModel):
    """
    Context summary used to derive reasonable default interpolation parameters.

    Attributes:
        data_scale: Characteristic spatial scales in (x, y, z) derived from point cloud spread.
        n_points: Number of constraint points.
        mean_nn_distance: Median/mean nearest-neighbour distance (used as a characteristic length).
    """

    data_scale: tuple[float, float, float]
    n_points: int
    mean_nn_distance: float


class OrdinaryKrigingParams(BaseModel):
    """
    Configuration parameters for Ordinary Kriging interpolation.

    Notes:
        This model is used as a parameter container; changing defaults affects only
        configuration values, not the interpolation algorithm implementation here.

    Attributes:
        variogram_model: Variogram model type (e.g. "spherical", "exponential", "gaussian").
        range: Range of the variogram model.
        sill: Sill of the variogram model (max variance level).
        nugget: Nugget effect.
        anisotropy_scaling_x: Axis scaling for anisotropy handling.
        anisotropy_scaling_y: Axis scaling for anisotropy handling.
        anisotropy_scaling_z: Axis scaling for anisotropy handling.
        neighbors: Optional number of nearest neighbors. If None, all points are used.
    """

    variogram_model: str = Field(
        "gaussian",
        description="Type of variogram model (e.g., spherical, exponential, gaussian).",
    )
    range: float = Field(
        500.0,
        description="Range of the variogram (distance at which correlation tapers off).",
    )
    sill: float = Field(1.0, description="Sill of the variogram (max variance level).")
    nugget: float = Field(0.0, description="Nugget effect (variance at zero distance).")
    anisotropy_scaling_x: float = Field(
        1.0, description="Scaling factor for the x-axis in 3D kriging."
    )
    anisotropy_scaling_y: float = Field(
        1.0, description="Scaling factor for the y-axis in 3D kriging."
    )
    anisotropy_scaling_z: float = Field(
        1.0, description="Scaling factor for the z-axis in 3D kriging."
    )
    neighbors: Optional[int] = Field(
        None,
        description="Number of nearest neighbors to use in kriging. If None, uses all points.",
    )


def default_ok_params(ctx: InterpolationContext) -> OrdinaryKrigingParams:
    """
    Derive heuristic default Ordinary Kriging parameters from an interpolation context.

    Args:
        ctx: Context statistics (data scale, point count, NN distance).

    Returns:
        A populated :class:`OrdinaryKrigingParams`.
    """
    sx, sy, sz = ctx.data_scale
    npts = ctx.n_points
    nn_dist = ctx.mean_nn_distance

    # Variogram: gaussian as default for smoother residuals; can be changed to exponential or others if needed
    variogram_model = "gaussian"

    # Range
    range_ = np.clip(
        20 * nn_dist,
        0.1 * max(sx, sy, sz),
        0.8 * max(sx, sy, sz),
    )

    sill = 1.0
    nugget = 0.0

    # Anisotropy scaling per axis (scale relative to largest dimension)
    max_scale = max(sx, sy, sz)
    anisotropy_scaling_x = 1.0
    anisotropy_scaling_y = np.clip(sy / sx, 0.05, 1.0)
    anisotropy_scaling_z = np.clip(sz / sx, 0.05, 1.0)

    # Rotation angles (degrees)
    # Default 0 → no rotation, but could be adapted if you detect tilted layers
    anisotropy_angle_x = 0.0
    anisotropy_angle_y = 0.0
    anisotropy_angle_z = 0.0
    # (angles currently not used in this file; kept as documentation of intent)

    # Neighbors (moving window)
    if npts < 20:
        neighbors = None  # global kriging
    else:
        neighbors = min(200, max(30, npts // 10))

    return OrdinaryKrigingParams(
        variogram_model=variogram_model,
        range=range_,
        sill=sill,
        nugget=nugget,
        anisotropy_scaling_x=float(anisotropy_scaling_x),
        anisotropy_scaling_y=float(anisotropy_scaling_y),
        anisotropy_scaling_z=float(anisotropy_scaling_z),
        neighbors=neighbors,
    )


class RBFParams(BaseModel):
    """
    Parameters for Radial Basis Function (RBF) interpolation.

    Attributes:
        kernel: RBF kernel (e.g., 'linear', 'cubic', 'thin_plate_spline', etc.).
        smoothing: Smoothing parameter (0 means exact fit).
        epsilon: Shape parameter for certain kernels (multiquadric, inverse multiquadric, gaussian).
        neighbors: Optional number of nearest neighbors. If None, all points are used.
    """

    kernel: str = Field(
        "thin_plate_spline",
        description="Radial basis function kernel. Common options: 'linear', 'cubic', 'thin_plate'.",
    )
    smoothing: float = Field(
        0,
        description="Smoothing parameter for RBF. Higher values increase smoothing (0 = exact fit).",
    )
    epsilon: Optional[float] = Field(
        None,
        description="Shape parameter for certain kernels like multiquadric or inverse multiquadric.",
    )
    neighbors: Optional[int] = Field(
        None,
        description="Number of nearest neighbors to use. If None, all points are used.",
    )


def default_rbf_params(ctx: InterpolationContext) -> RBFParams:
    """
    Derive heuristic default RBF parameters from an interpolation context.

    Args:
        ctx: Context statistics (data scale, point count, NN distance).

    Returns:
        A populated :class:`RBFParams`.
    """
    # Characteristic length scale (currently informational; may influence future defaults)
    cl = max(ctx.data_scale)

    kernel = "thin_plate_spline"

    epsilon: Optional[float] = None
    if kernel in {"multiquadric", "inverse_multiquadric", "gaussian"}:
        epsilon = ctx.mean_nn_distance

    # Neighbors scale with problem size
    neighbors: Optional[int] = None
    if ctx.n_points > 5000:
        neighbors = min(500, int(5 * ctx.n_points ** (2 / 3)))

    return RBFParams(
        kernel="thin_plate_spline",
        smoothing=0.05,  # categorical scalar field → smooth by default
        epsilon=epsilon,
        neighbors=neighbors,
    )


GeoINRActivation = Literal["Softplus", "ReLU", "LeakyReLU", "Tanh", "Sigmoid", "ELU", "PReLU"]


class GeoINRParams(BaseModel):
    """
    Parameters controlling GeoINR interpolation (implicit neural representation).

    This parameter set maps directly to the arguments used by your GeoINR training routine
    `stratigraphic_ConcatMLP()` and the network `ConcatMLP`.

    Notes (current implementation)
    ------------------------------
    - Coordinates are normalized into [-1, 1] using the provided model extent.
    - Interface constraints are trained with an L1 loss on predicted scalar vs. label.
    - Orientation constraints are trained by matching the gradient direction of the scalar field
      to the provided orientation vectors (cosine-based loss).
    """

    # --- core "geological" weights ---
    beta: float = Field(
        1.0,
        ge=0.01,
        description=(
            "Softplus beta parameter when activation='Softplus'. Higher beta makes Softplus "
            "closer to ReLU (sharper transitions). In your current code this is the only "
            "group-level parameter that is used."
        ),
    )

    alpha: float = Field(
        0.1,
        ge=0.0,
        description=(
            "Weight of the orientation (gradient) loss relative to the interface loss. "
            "Total loss = loss_interface + alpha * loss_orientation. Increase alpha to "
            "enforce structural orientations more strongly; decrease if interfaces dominate "
            "or orientations are noisy/sparse."
        ),
    )

    # --- network capacity / shape ---
    hidden_dim: int = Field(
        32,
        ge=4,
        description=(
            "Width of hidden layers in the MLP. Larger values increase representational "
            "capacity (can fit more complex folding/faulting) but increase runtime and "
            "overfitting risk."
        ),
    )

    n_hidden_layers: int = Field(
        1,
        ge=0,
        description=(
            "Number of hidden layers (excluding input and output layers). More layers "
            "increase model capacity; start small (1–3) and scale up for complex geometry."
        ),
    )

    # activation: GeoINRActivation = Field(
    #     "Softplus",
    #     description=(
    #         "Activation function used by the MLP. 'Softplus' is a smooth default that tends "
    #         "to produce smooth scalar fields; ReLU/LeakyReLU can create sharper features."
    #     ),
    # )

    # concat: bool = Field(
    #     False,
    #     description=(
    #         "If True, concatenate input coordinates with hidden features at each layer "
    #         "(a skip/concat architecture). This can improve expressivity but grows layer "
    #         "sizes quickly and can become expensive."
    #     ),
    # )

    # --- training / optimization ---
    epochs: int = Field(
        5000,
        ge=1,
        description=(
            "Number of training epochs for the INR. Higher values can improve fit but "
            "increase runtime. Consider fewer epochs for large datasets or when iterating."
        ),
    )

    lr: float = Field(
        0.01,
        gt=0.0,
        description=(
            "Learning rate for AdamW optimizer. If training is unstable (loss oscillates), "
            "reduce lr (e.g., 0.003 or 0.001)."
        ),
    )

    # weight_decay: float = Field(
    #     0.0,
    #     ge=0.0,
    #     description=(
    #         "AdamW weight decay (L2 regularization) applied to network weights. Useful to "
    #         "reduce overfitting, especially with very sparse constraints or high-capacity models."
    #     ),
    # )

    # --- reproducibility / runtime knobs (optional, but handy) ---
    # seed: Optional[int] = Field(
    #     None,
    #     description=(
    #         "Random seed for reproducible training. If None, training is stochastic "
    #         "(GPU + PyTorch nondeterminism may still apply)."
    #     ),
    # )

    # verbose: bool = Field(
    #     True,
    #     description=(
    #         "If True, print training/inference diagnostics (losses, times). "
    #         "Set False to silence routine output."
    #     ),
    # )


def default_geo_inr_params(ctx: InterpolationContext) -> GeoINRParams:
    """
    Heuristic defaults for GeoINR based on problem size.

    Heuristic intent
    ---------------
    - Small datasets: slightly higher epochs and/or lr to converge quickly.
    - Large datasets: reduce epochs and lr, increase hidden_dim modestly.
    - Keep architecture conservative by default (avoids overfitting and long runtimes).
    """

    n = ctx.n_points

    # Capacity scaling (conservative)
    if n < 500:
        hidden_dim = 32
        n_hidden_layers = 2
    elif n < 3000:
        hidden_dim = 32
        n_hidden_layers = 1
    else:
        hidden_dim = 64
        n_hidden_layers = 2

    # Epoch scaling (sublinear-ish)
    # - small: more epochs are cheap and often needed for smooth INR convergence
    # - large: reduce epochs for runtime
    if n < 800:
        epochs = 6000
        lr = 0.01
    elif n < 5000:
        epochs = 4000
        lr = 0.005
    else:
        epochs = 2500
        lr = 0.003

    # Orientation weight:
    # Without knowing orientation count/quality, keep moderate.
    alpha = 0.1

    # Softplus beta:
    # Keep at 1.0 for smoothness; bump slightly for sharper transitions if desired.
    beta = 1.0

    # Mild regularization for larger problems
    weight_decay = 0.0 if n < 3000 else 1e-4

    return GeoINRParams(
        beta=beta,
        alpha=alpha,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        activation="Softplus",
        # concat=False,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        # seed=None,
        # verbose=False,
    )


class LoopStructuralMethod(str, Enum):
    """LoopStructural interpolator selection."""

    FDI = "FDI"
    PLI = "PLI"


class LoopStructuralParams(BaseModel):
    """
    Parameters for LoopStructural interpolation.

    Attributes:
        interpolator_type: Choice of interpolator in LoopStructural.
    """

    interpolator_type: LoopStructuralMethod = Field(
        default=LoopStructuralMethod.FDI,
        description="Type of LoopStructural interpolator. Choose 'FDI' or 'PLI'.",
    )


class UniversalCoKrigingParams(BaseModel):
    """
    Placeholder class for Universal Co-Kriging interpolation parameters.

    Currently, Universal Co-Kriging does not require any parameters,
    but this class is in place to support future configuration needs.
    """

    pass


class UniversalKrigingParams(OrdinaryKrigingParams):
    """
    Configuration parameters for Universal Kriging interpolation.

    Adds drift (trend) configuration on top of ordinary kriging parameters.
    """

    drift_terms: Union[str, List[str]] = Field(
        "regional_linear",
        description=(
            "Drift terms for Universal Kriging. Common: 'regional_linear'. "
            "Can be a string or list of strings."
        ),
    )


def default_uk_params(ctx: InterpolationContext) -> UniversalKrigingParams:
    """
    Derive heuristic default Universal Kriging parameters from an interpolation context.

    UK defaults are similar to OK, but:
      - drift_terms defaults to "regional_linear" (good for geology),
      - range is slightly larger (UK residual field tends to be smoother),
      - anisotropy scaling uses ratios relative to X (PyKrige convention).
    """
    sx, sy, sz = ctx.data_scale
    npts = ctx.n_points
    nn_dist = ctx.mean_nn_distance

    # Variogram: gaussian as default for smoother residuals; can be changed to exponential or others if needed
    variogram_model = "gaussian"

    # Range: a bit longer than OK
    max_scale = max(sx, sy, sz)
    range_ = np.clip(
        30 * nn_dist,  # OK used 20*nn_dist; UK often benefits from longer
        0.15 * max_scale,
        0.9 * max_scale,
    )

    sill = 1.0
    nugget = 0.0

    # Anisotropy scaling (PyKrige 3D: x is implicit; scale y/z relative to x)
    # Guard against sx ~ 0
    sx_safe = max(float(sx), 1e-12)
    anisotropy_scaling_x = 1.0
    anisotropy_scaling_y = float(np.clip(sy / sx_safe, 0.05, 20.0))
    anisotropy_scaling_z = float(np.clip(sz / sx_safe, 0.05, 20.0))

    # Neighbors (moving window)
    if npts < 20:
        neighbors = None
    else:
        neighbors = min(200, max(30, npts // 10))

    return UniversalKrigingParams(
        variogram_model=variogram_model,
        range=float(range_),
        sill=float(sill),
        nugget=float(nugget),
        anisotropy_scaling_x=float(anisotropy_scaling_x),
        anisotropy_scaling_y=float(anisotropy_scaling_y),
        anisotropy_scaling_z=float(anisotropy_scaling_z),
        neighbors=neighbors,
        drift_terms="regional_linear",
    )


InterpolationParameterSet = Union[
    OrdinaryKrigingParams,
    RBFParams,
    GeoINRParams,
    LoopStructuralParams,
    UniversalCoKrigingParams,
    UniversalKrigingParams,
]
