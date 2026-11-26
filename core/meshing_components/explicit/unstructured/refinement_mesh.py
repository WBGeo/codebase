from typing import List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from py_api_wbgeo.nodesapi import wbgeo_type
from core import workbench_palette as palette

# ------------------------
# Wells
# ------------------------
@wbgeo_type(name="Linear Well Refinement", identifier="LinearWellRefinement", color=palette.PARAMETER)
class LinearWellRefinement(BaseModel):
    """
    Refines the mesh around wells using a distance-based linear
    interpolation between a minimum and maximum element size.
    """
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float


@wbgeo_type(name="Function Well Refinement", identifier="FunctionWellRefinement", color=palette.PARAMETER)
class FunctionWellRefinement(BaseModel):
    """
    Refines the mesh around wells using a user-defined Gmsh
    mathematical expression.
    """
    expression: str


# ------------------------
# Sources
# ------------------------
@wbgeo_type(name="Linear Source Refinement", identifier="LinearSourceRefinement", color=palette.PARAMETER)
class LinearSourceRefinement(BaseModel):
    """
    Refines the mesh around source locations using distance-based linear scaling.
    """
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float


@wbgeo_type(name="Function Source Refinement", identifier="FunctionSourceRefinement", color=palette.PARAMETER)
class FunctionSourceRefinement(BaseModel):
    """
    Refines mesh around sources using a Gmsh mathematical expression.
    """
    expression: str


# ------------------------
# Mine shafts
# ------------------------
@wbgeo_type(
    name="Linear Shaft Refinement",
    identifier="LinearShaftRefinement",
    color=palette.PARAMETER,
)
class LinearShaftRefinement(BaseModel):
    """
    Refines the mesh according to perpendicular distance from
    the axis of the nearest mine shaft.
    """
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float


@wbgeo_type(
    name="Function Shaft Refinement",
    identifier="FunctionShaftRefinement",
    color=palette.PARAMETER,
)
class FunctionShaftRefinement(BaseModel):
    """
    User-defined mesh-size expression based on perpendicular
    distance from the nearest mine-shaft axis.

    Use DIST in the expression for the axis distance.
    """
    expression: str


# ------------------------
# Triangulation
# ------------------------
@wbgeo_type(name="Triangulation Refinement", identifier="TriangulationRefinement", color=palette.PARAMETER)
class TriangulationRefinement(BaseModel):
    """
    Mesh refinement around triangulated surfaces.
    """
    SizeMin: float = 3.0
    SizeMax: float = 30.0
    DistMin: float = 20.0
    DistMax: float = 60.0


# ------------------------
# Ellipses
# ------------------------
@wbgeo_type(name="Ellipse Refinement", identifier="EllipseRefinement", color=palette.PARAMETER)
class EllipseRefinement(BaseModel):
    """
    Mesh refinement around elliptical regions.
    """
    SizeMin: float = 3.0
    SizeMax: float = 30.0
    DistMin: float = 20.0
    DistMax: float = 60.0


# ------------------------
# Faults
# ------------------------
@wbgeo_type(name="Fault Refinement", identifier="FaultRefinement", color=palette.PARAMETER)
class FaultRefinement(BaseModel):
    """
    Mesh refinement around fault surfaces.
    """
    SizeMin: float = 3.0
    SizeMax: float = 30.0
    DistMin: float = 20.0
    DistMax: float = 60.0


@wbgeo_type(name="Extra Plane Refinement",
    identifier="ExtraPlaneRefinement",
    color=palette.PARAMETER,
)
class ExtraPlaneRefinement(BaseModel):
    """
    Mesh refinement around extra planes using
    distance-based linear scaling.
    """
    SizeMin: float = 3.0
    SizeMax: float = 30.0
    DistMin: float = 20.0
    DistMax: float =60.0

Coordinate3D = Tuple[float, float, float]


@wbgeo_type(
    name="General Refinement",
    identifier="GeneralRefinement",
    color=palette.PARAMETER,
)
class GeneralRefinement(BaseModel):
    """
    General point- or line-based refinement.

    One coordinate:
        coordinates=[(x, y, z)]

        DIST is the Euclidean distance from that point.

    Two coordinates:
        coordinates=[
            (x1, y1, z1),
            (x2, y2, z2),
        ]

        DIST is the distance from the finite line segment connecting
        the two coordinates.

    Linear refinement:
        Set SizeMin and DistMax.

    Function refinement:
        Set expression and use DIST inside the expression.
    """
    coordinates: List[Coordinate3D]

    SizeMin: Optional[float] = None
    DistMax: Optional[float] = None
    expression: Optional[str] = None

# ------------------------
# MAIN CONTAINER
# ------------------------
@wbgeo_type(name="Mesh Refinement", identifier="Refinement", color=palette.PARAMETER)
class Refinement(BaseModel):
    """Configuration of optional mesh-refinement regions and strategies.

    Each field enables refinement around the corresponding geometric feature.
    Linear strategies specify element sizes and distance thresholds, while
    function strategies use a Gmsh mesh-size expression. The ``general`` field
    accepts one or multiple point- or line-based refinements.

    ``extra_planes`` is the preferred Python attribute name. The legacy
    ``extraplanes`` input name remains accepted for compatibility.
    """

    class Config:
        allow_population_by_field_name = True  # Pydantic v1
        populate_by_name = True  # Pydantic v2

    wells: Optional[
        Union[LinearWellRefinement, FunctionWellRefinement]
    ] = None

    sources: Optional[
        Union[LinearSourceRefinement, FunctionSourceRefinement]
    ] = None

    shafts: Optional[
        Union[LinearShaftRefinement, FunctionShaftRefinement]
    ] = None

    # One or more general point/line refinements
    general: Optional[
        Union[
            GeneralRefinement,
            List[GeneralRefinement],
        ]
    ] = None

    triangulations: Optional[TriangulationRefinement] = None
    ellipses: Optional[EllipseRefinement] = None
    extra_planes: Optional[ExtraPlaneRefinement] = Field(default=None, alias="extraplanes")

    @property
    def extraplanes(self) -> Optional[ExtraPlaneRefinement]:
        """Legacy attribute alias for :attr:`extra_planes`."""
        return self.extra_planes
    faults: Optional[FaultRefinement] = None
