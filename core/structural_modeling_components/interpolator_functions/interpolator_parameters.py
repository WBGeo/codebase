from __future__ import annotations

from enum import Enum
from typing import Dict, FrozenSet, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, PrivateAttr

# -----------------------------------------------------------------------------
# Type aliases (readability only)
# -----------------------------------------------------------------------------
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
BoolArray = npt.NDArray[np.bool_]

MeshDict = Dict[str, npt.NDArray[np.generic]]


class InterpolationMethod(str, Enum):
    """Supported interpolation backends for structural scalar fields."""

    ORDINARY_KRIGING = "Ordinary Kriging"
    RADIAL_BASIS_FUNCTION = "Radial Basis Function"
    UNIVERSAL_COKRIGING = "Universal Co-Kriging"
    GEOINR = "GeoINR"
    FINITE_DIFFERENCES = "Finite Differences"
    PIECEWISE_LINEAR = "Piecewise Linear"
    UNIVERSAL_KRIGING = "Universal Kriging"


class VariogramModel(str, Enum):
    """Variogram models available for Ordinary and Universal Kriging.

    PyKrige's "linear" and "power" models are left out: they take a different
    parameter list than the range/nugget set exposed here.
    """

    GAUSSIAN = "gaussian"
    SPHERICAL = "spherical"
    EXPONENTIAL = "exponential"
    HOLE_EFFECT = "hole-effect"


class RBFKernel(str, Enum):
    """Kernels available for Radial Basis Function interpolation (SciPy ``RBFInterpolator``)."""

    THIN_PLATE_SPLINE = "thin_plate_spline"
    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    INVERSE_QUADRATIC = "inverse_quadratic"
    GAUSSIAN = "gaussian"


# Kernels whose shape depends on `epsilon`. The remaining kernels are scale-invariant
# and are evaluated without it.
RBF_KERNELS_USING_EPSILON: FrozenSet[str] = frozenset(
    {
        RBFKernel.MULTIQUADRIC.value,
        RBFKernel.INVERSE_MULTIQUADRIC.value,
        RBFKernel.INVERSE_QUADRATIC.value,
        RBFKernel.GAUSSIAN.value,
    }
)


# Enum-typed fields are stored as their plain string value, so interpolators and
# comparisons keep working with strings; assignments are validated so that an
# unknown option is rejected when it is set, not later inside the backend library.
_ENUM_FIELD_CONFIG = ConfigDict(use_enum_values=True, validate_default=True, validate_assignment=True)


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


class _KrigingParams(BaseModel):
    """
    Variogram and anisotropy parameters shared by Ordinary and Universal Kriging.

    The sill is fixed at 1: kriging results depend only on the shape of the
    variogram (model, range) and on the nugget relative to the sill, not on the
    sill itself. A tiny jitter (``KRIGING_NUGGET_JITTER``) is always added to the
    nugget to keep the kriging matrix well conditioned (most important for the
    Gaussian variogram); it is not visible in the result.

    Attributes:
        variogram_model: Variogram model type, one of :class:`VariogramModel`.
        range: Range of the variogram model (distance, in model units).
        nugget: Nugget as a fraction of the sill (0 = exact fit through the points).
        anisotropy_scaling_y: Factor applied to y coordinates (relative to x).
        anisotropy_scaling_z: Factor applied to z coordinates (relative to x).
    """

    model_config = _ENUM_FIELD_CONFIG

    variogram_model: VariogramModel = Field(
        VariogramModel.GAUSSIAN,
        description="Type of variogram model. 'gaussian' gives the smoothest surfaces.",
    )
    range: float = Field(
        500.0,
        gt=0.0,
        description="Range of the variogram: distance (in model units) up to which points influence each other.",
    )
    nugget: float = Field(
        0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Nugget as a fraction of the sill (0 to below 1). 0 = surfaces pass exactly through the "
            "points; larger values allow deviations and give smoother surfaces, e.g. for noisy data."
        ),
    )
    anisotropy_scaling_y: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Factor applied to y coordinates before kriging (1 = no anisotropy). Values below 1 "
            "make distances along y count less, i.e. a longer range in y."
        ),
    )
    anisotropy_scaling_z: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Factor applied to z coordinates before kriging (1 = no anisotropy). Values below 1 "
            "make vertical distances count less, i.e. a longer range in z."
        ),
    )


# Added to every nugget (relative to the sill). Chosen on the model 2 example: without it,
# the Gaussian variogram gives a numerically degraded field (95.2 % agreement with RBF,
# 43.8 % without moving window); 1e-6 gives 97.6 % (95.4 % without moving window).
KRIGING_NUGGET_JITTER: float = 1e-6


def kriging_variogram_parameters(params: "_KrigingParams") -> List[float]:
    """PyKrige ``variogram_parameters`` [sill, range, nugget] with sill 1 and the jitter added."""
    return [1.0, float(params.range), float(params.nugget) + KRIGING_NUGGET_JITTER]


class OKParams(_KrigingParams):
    """
    Configuration parameters for Ordinary Kriging interpolation.

    Attributes (in addition to :class:`_KrigingParams`):
        neighbors: Optional number of nearest neighbors (moving window). If None, all points are used.
    """

    neighbors: Optional[int] = Field(
        None,
        ge=1,
        description=(
            "Number of nearest points used at each grid cell (moving window). If empty, all points "
            "are used. If it is not smaller than the number of points, all points are used."
        ),
    )


def _kriging_anisotropy_defaults(ctx: InterpolationContext) -> Tuple[float, float]:
    """
    Anisotropy scaling from the spread of the points: each axis is scaled relative to x by
    its spread, so the variogram range is effectively longer along axes with little spread
    (typically z for layered data). Better than isotropic scaling on the model 2 example.
    """
    sx, sy, sz = ctx.data_scale
    sx_safe = max(float(sx), 1e-12)
    return (float(np.clip(sy / sx_safe, 0.05, 1.0)), float(np.clip(sz / sx_safe, 0.05, 1.0)))


def default_ok_params(ctx: InterpolationContext) -> OKParams:
    """
    Derive heuristic default Ordinary Kriging parameters from an interpolation context.

    Args:
        ctx: Context statistics (data scale, point count, NN distance).

    Returns:
        A populated :class:`OKParams`.
    """
    sx, sy, sz = ctx.data_scale
    npts = ctx.n_points
    nn_dist = ctx.mean_nn_distance

    # Range: use the data diagonal as the reference scale so that the range
    # covers the full data extent even when points are sparse at corners.
    # (Using only max(sx,sy,sz) would underestimate the range for data spread
    # across multiple axes, causing a "pillow" artefact with 4-corner data.)
    data_diagonal = float(np.sqrt(sx ** 2 + sy ** 2 + sz ** 2))
    range_ = np.clip(
        20 * nn_dist,
        0.1 * max(sx, sy, sz),
        1.5 * data_diagonal,
    )

    anisotropy_scaling_y, anisotropy_scaling_z = _kriging_anisotropy_defaults(ctx)

    # Neighbors (moving window)
    if npts < 20:
        neighbors = None  # global kriging
    else:
        neighbors = min(200, max(30, npts // 10))

    return OKParams(
        variogram_model=VariogramModel.GAUSSIAN,
        range=float(range_),
        nugget=0.0,
        anisotropy_scaling_y=anisotropy_scaling_y,
        anisotropy_scaling_z=anisotropy_scaling_z,
        neighbors=neighbors,
    )


class RBFParams(BaseModel):
    """
    Parameters for Radial Basis Function (RBF) interpolation.

    Attributes:
        kernel: RBF kernel, one of :class:`RBFKernel`.
        smoothing: Smoothing parameter (0 means exact fit).
        epsilon: Shape parameter (inverse length). Only used by the kernels in
            ``RBF_KERNELS_USING_EPSILON``; ignored for the others.
        neighbors: Optional number of nearest neighbors. If None, all points are used.
        anisotropy_scaling_z: Factor applied to all z coordinates (points and grid)
            before fitting. 1.0 means no anisotropy.
    """

    model_config = _ENUM_FIELD_CONFIG

    kernel: RBFKernel = Field(
        RBFKernel.THIN_PLATE_SPLINE,
        description="Radial basis function kernel.",
    )
    smoothing: float = Field(
        0,
        description="Smoothing parameter for RBF. Higher values increase smoothing (0 = exact fit).",
    )
    epsilon: Optional[float] = Field(
        None,
        description=(
            "Shape parameter (inverse length; larger = more local). Only used by the kernels "
            "multiquadric, inverse_multiquadric, inverse_quadratic and gaussian; ignored for the others."
        ),
    )
    neighbors: Optional[int] = Field(
        None,
        description="Number of nearest neighbors to use. If None, all points are used.",
    )
    anisotropy_scaling_z: float = Field(
        1.0,
        gt=0.0,
        description=(
            "Vertical stretching factor (1 = no anisotropy). Values above 1 make vertical "
            "distances count more than horizontal ones, so points influence each other more "
            "along the layers than across them. Values below 1 do the opposite. epsilon refers "
            "to the stretched coordinates."
        ),
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

    # Always provide a shape parameter so that switching to a kernel that needs one
    # works without further input. SciPy scales distances by epsilon, so it is an
    # inverse length: a few nearest-neighbour distances per kernel width.
    nn_dist = ctx.mean_nn_distance if ctx.mean_nn_distance > 0 else max(cl, 1.0)
    epsilon = 1.0 / (5.0 * nn_dist)

    # Neighbors scale with problem size
    neighbors: Optional[int] = None
    if ctx.n_points > 5000:
        neighbors = min(500, int(5 * ctx.n_points ** (2 / 3)))

    return RBFParams(
        kernel=RBFKernel.THIN_PLATE_SPLINE,
        smoothing=0.05,  # categorical scalar field → smooth by default
        epsilon=epsilon,
        neighbors=neighbors,
        anisotropy_scaling_z=1.0,
    )


class GeoINRParams(BaseModel):
    """
    Parameters of GeoINR interpolation (implicit neural representation).

    A small neural network (multilayer perceptron) is trained to output the scalar field
    at any location. Its loss combines three terms: the field value at the surface points
    (each formation has a fixed label between 1 and -1), the field's gradient direction
    at the orientations (weight ``orientation_weight``), and a regularisation that keeps
    the gradient magnitude similar everywhere in the model (fixed weight).

    Training starts from random network weights; ``seed`` fixes them, so the same input
    gives the same model.

    Fixed (not exposed): Softplus activation with beta 1, no skip connections.

    Older saved options may contain ``alpha`` (the previous name of
    ``orientation_weight``, still accepted) and ``beta`` (now fixed, ignored).
    """

    orientation_weight: float = Field(
        0.1,
        ge=0.0,
        validation_alias=AliasChoices("orientation_weight", "alpha"),
        description=(
            "How strongly orientations count compared with surface points. Larger values make "
            "the layers follow the measured dips more closely; smaller values let the surfaces "
            "follow the surface points more closely."
        ),
    )

    # --- network capacity / shape ---
    hidden_dim: int = Field(
        32,
        ge=4,
        description=(
            "Number of neurons per hidden layer of the network. More neurons can represent "
            "more complex geometry (e.g. tight folds) but train more slowly and may overfit "
            "sparse data."
        ),
    )

    n_hidden_layers: int = Field(
        1,
        ge=0,
        description=(
            "Number of hidden layers of the network. More layers can represent more complex "
            "geometry; 1 to 3 is usually enough."
        ),
    )

    # --- training ---
    epochs: int = Field(
        5000,
        ge=1,
        description=(
            "Number of training steps. More steps can fit the data better but take longer."
        ),
    )

    lr: float = Field(
        0.01,
        gt=0.0,
        description=(
            "Learning rate: step size of the training. If the result is unstable or "
            "differs strongly between seeds, try a smaller value (e.g. 0.003 or 0.001)."
        ),
    )

    seed: int = Field(
        0,
        ge=0,
        description=(
            "Start value of the random number generator. Training starts from random network "
            "weights; the same seed gives the same model. Try a few other seeds to see how "
            "much the result depends on chance."
        ),
    )


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
    # n<800 case: validated against a real wbgeo model (84/36-point groups, one with a sharp
    # local fold) at 3000/4000/6000 epochs - all three were equally reliable (10/10 clean
    # across random seeds), so lowered from 6000 to 4000 for a ~30% runtime reduction with no
    # observed reliability cost. Only tested on that one dataset - revisit with a proper
    # input-data-driven calibration (like the other interpolators' heuristics) rather than
    # assuming this generalizes to every dataset in this size bucket.
    if n < 800:
        epochs = 4000
        lr = 0.01
    elif n < 5000:
        epochs = 4000
        lr = 0.005
    else:
        epochs = 2500
        lr = 0.003

    # Orientation weight:
    # Without knowing orientation count/quality, keep moderate.
    orientation_weight = 0.1

    return GeoINRParams(
        orientation_weight=orientation_weight,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        epochs=epochs,
        lr=lr,
    )


class FDIParams(BaseModel):
    """
    Parameters for LoopStructural FDI (finite difference) interpolation.

    FDI solves the scalar field on its own regular cartesian support (finer with larger
    ``nelements``) as a weighted least-squares problem: surface points (weight 1),
    orientations (``orientation_weight``) and smoothness (``regularisation``). Only the
    ratios between the three weights matter. The solver is fixed to conjugate gradients
    with LoopStructural's default tolerance.

    Attributes
    ----------
    nelements:
        Number of elements of LoopStructural's support (larger = finer, slower).
    regularisation:
        Weight of the smoothness term. Default 0.1 instead of LoopStructural's 1.0: on
        synthetic model 2 it raised the agreement with the RBF model from 88.5 % to 95.6 %
        of the cells, whereas a finer support (more ``nelements``) gained at most about 2 %
        at many times the runtime.
    orientation_weight:
        Weight of the orientation constraints relative to the surface points.
    """

    nelements: int = Field(
        10_000,
        ge=1_000,
        description=(
            "Number of elements of the computation grid LoopStructural uses internally "
            "(larger = finer surfaces, slower)."
        ),
    )
    regularisation: float = Field(
        0.1,
        gt=0.0,
        description=(
            "Smoothness of the scalar field. Larger values give smoother, more planar surfaces "
            "that may not honour every data point; smaller values follow the data more closely."
        ),
    )
    orientation_weight: float = Field(
        1.0,
        gt=0.0,
        description=(
            "How strongly orientations count compared with surface points (1 = equally). Larger "
            "values make the layers follow the measured dips more closely; smaller values let the "
            "surfaces follow the surface points more closely."
        ),
    )


def default_fdi_params(ctx: "InterpolationContext") -> FDIParams:
    """
    Derive heuristic default LoopStructural FDI parameters from an interpolation context.

    Heuristics
    ----------
    - `nelements` grows sublinearly with problem size: more data -> allow finer support,
      but keep a hard cap to avoid runaway compute/memory costs.

    Returns
    -------
    FDIParams
        A populated parameter object.
    """
    # Start from LoopStructural's common default nelements (~1e4) and scale with point count.
    # Use sqrt scaling so it increases, but gently:
    #   1k pts -> ~10k
    #   4k pts -> ~20k
    #   25k pts -> ~50k
    #
    # Clamp to keep memory/runtime predictable.
    base = 10_000
    scale = (max(ctx.n_points, 1) / 1_000) ** 0.5
    nelements = int(base * max(0.7, min(scale, 8.0)))  # factor in [0.7 .. 8.0]
    nelements = max(5_000, min(nelements, 200_000))

    return FDIParams(
        nelements=nelements,
    )


class PLIParams(BaseModel):
    """
    Parameters for LoopStructural PLI (piecewise linear) interpolation.

    PLI solves the scalar field on a tetrahedral support (finer with larger ``nelements``)
    with the same weighted least-squares terms as FDI.

    In LoopStructural 1.8.2 the gradient constraint of the piecewise-linear interpolator
    is not implemented, so orientations are passed as tangent constraints (strike and dip
    vectors), weighted by ``orientation_weight``. LoopStructural scales these with the
    element volume; the interpolator divides the weight by the mean element volume, so
    it does not depend on ``nelements``.

    Attributes
    ----------
    nelements:
        Number of elements of LoopStructural's support (larger = finer, slower).
    regularisation:
        Weight of the smoothness term (constant-gradient regularisation;
        LoopStructural default for PLI: 0.1).
    orientation_weight:
        Weight of the orientation (tangent) constraints relative to the surface points.
    """

    nelements: int = Field(
        2_000,
        ge=1_000,
        description=(
            "Number of elements of the computation mesh LoopStructural uses internally "
            "(larger = finer surfaces, slower)."
        ),
    )
    regularisation: float = Field(
        0.1,
        gt=0.0,
        description=(
            "Smoothness of the scalar field. Larger values give smoother, more planar surfaces "
            "that may not honour every data point; smaller values follow the data more closely."
        ),
    )
    orientation_weight: float = Field(
        1.0,
        gt=0.0,
        description=(
            "How strongly orientations count compared with surface points (1 = equally). Larger "
            "values make the layers follow the measured dips more closely; smaller values let the "
            "surfaces follow the surface points more closely."
        ),
    )


def default_pli_params(ctx: "InterpolationContext") -> PLIParams:
    """
    Derive heuristic default LoopStructural PLI parameters from an interpolation context.

    Heuristics
    ----------
    - PLI's unstructured tetrahedral mesh is much more expensive per element than FDI's
      structured grid, so we start from a lower base `nelements` than FDI.
    - `nelements` grows sublinearly with problem size (sqrt scaling with point count),
      and is softly adjusted by how well-sampled the domain is (mean NN distance vs extent).

    Returns
    -------
    PLIParams
        A populated parameter object.
    """
    # --- base scaling with problem size ---
    # PLI builds an unstructured tetrahedral mesh which is much more expensive per element
    # than FDI's structured Cartesian grid.  A lower base keeps runtimes reasonable for
    # small/medium inputs while still scaling up for large production datasets.
    base = 2_000

    # Sublinear growth with number of constraints.
    scale_n = (max(ctx.n_points, 1) / 1_000) ** 0.5  # gentle growth

    # --- sampling-density adjustment (optional but useful) ---
    # If points are dense relative to the domain extent, we can afford (and benefit from)
    # a bit more discretisation; if points are sparse, keep it conservative.
    extent = max(ctx.data_scale)
    if extent > 0:
        # dimensionless: bigger means denser sampling
        density = extent / max(ctx.mean_nn_distance, 1e-9)
        # Map density into a mild multiplier in ~[0.8 .. 1.4]
        # (log keeps it stable across huge ranges)
        import math

        scale_d = 0.8 + 0.1 * min(6.0, max(0.0, math.log10(max(density, 1.0))))
        scale_d = max(0.8, min(scale_d, 1.4))
    else:
        scale_d = 1.0

    nelements = int(base * max(0.8, min(scale_n, 10.0)) * scale_d)

    # Clamp: keep a low floor so tiny inputs don't over-allocate, and cap at 50k so
    # even very large models remain tractable without manual tuning.
    nelements = max(2_000, min(nelements, 50_000))

    return PLIParams(
        nelements=nelements,
    )


class UCKParams(BaseModel):
    """
    Placeholder class for Universal Co-Kriging interpolation parameters.

    Currently, Universal Co-Kriging does not require any parameters,
    but this class is in place to support future configuration needs.
    """

    pass


def default_uck_params(ctx: "InterpolationContext") -> UCKParams:
    # Placeholder default parameters for Universal Co-Kriging.
    return UCKParams()


class UKParams(_KrigingParams):
    """
    Configuration parameters for Universal Kriging interpolation.

    Same variogram and anisotropy parameters as Ordinary Kriging. A linear trend
    (PyKrige's "regional_linear" drift, a + b·x + c·y + d·z) is always fitted.
    PyKrige's 3D Universal Kriging has no moving window, so all points are used.
    """


def default_uk_params(ctx: InterpolationContext) -> UKParams:
    """
    Derive heuristic default Universal Kriging parameters from an interpolation context.

    UK defaults are similar to OK, but the range is slightly longer (the field left
    after removing the linear trend tends to be smoother) and the anisotropy scaling
    is allowed above 1.
    """
    sx, sy, sz = ctx.data_scale
    nn_dist = ctx.mean_nn_distance

    # Range: a bit longer than OK
    max_scale = max(sx, sy, sz)
    range_ = np.clip(
        30 * nn_dist,  # OK used 20*nn_dist; UK often benefits from longer
        0.15 * max_scale,
        0.9 * max_scale,
    )

    # Anisotropy scaling (PyKrige 3D: x is implicit; scale y/z relative to x)
    sx_safe = max(float(sx), 1e-12)
    anisotropy_scaling_y = float(np.clip(sy / sx_safe, 0.05, 20.0))
    anisotropy_scaling_z = float(np.clip(sz / sx_safe, 0.05, 20.0))

    return UKParams(
        variogram_model=VariogramModel.GAUSSIAN,
        range=float(range_),
        nugget=0.0,
        anisotropy_scaling_y=anisotropy_scaling_y,
        anisotropy_scaling_z=anisotropy_scaling_z,
    )


InterpolationParameterSet = Union[
    OKParams,
    RBFParams,
    GeoINRParams,
    FDIParams,
    PLIParams,
    UCKParams,
    UKParams,
]
