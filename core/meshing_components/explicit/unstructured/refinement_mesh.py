from typing import Union, Optional
from pydantic import BaseModel
from py_api_wbgeo.nodesapi import wbgeo_type


# ------------------------
# Wells
# ------------------------
@wbgeo_type(name="Linear Well Refinement", identifier="LinearWellRefinement")
class LinearWellRefinement(BaseModel):
    """
    Refines the mesh around wells using a distance-based linear
    interpolation between a minimum and maximum element size.
    """
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float


@wbgeo_type(name="Function Well Refinement", identifier="FunctionWellRefinement")
class FunctionWellRefinement(BaseModel):
    """
    Refines the mesh around wells using a user-defined Gmsh
    mathematical expression.
    """
    expression: str


# ------------------------
# Sources
# ------------------------
@wbgeo_type(name="Linear Source Refinement", identifier="LinearSourceRefinement")
class LinearSourceRefinement(BaseModel):
    """
    Refines the mesh around source locations using distance-based linear scaling.
    """
    SizeMin: float
    SizeMax: float
    DistMin: float
    DistMax: float


@wbgeo_type(name="Function Source Refinement", identifier="FunctionSourceRefinement")
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
@wbgeo_type(name="Triangulation Refinement", identifier="TriangulationRefinement")
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
@wbgeo_type(name="Ellipse Refinement", identifier="EllipseRefinement")
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
@wbgeo_type(name="Fault Refinement", identifier="FaultRefinement")
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

# ------------------------
# MAIN CONTAINER
# ------------------------
@wbgeo_type(name="Mesh Refinement", identifier="Refinement")
class Refinement(BaseModel):
    wells: Optional[
        Union[LinearWellRefinement, FunctionWellRefinement]
    ] = None

    sources: Optional[
        Union[LinearSourceRefinement, FunctionSourceRefinement]
    ] = None

    shafts: Optional[
        Union[LinearShaftRefinement, FunctionShaftRefinement]
    ] = None

    triangulations: Optional[TriangulationRefinement] = None
    ellipses: Optional[EllipseRefinement] = None
    extraplanes: Optional[ExtraPlaneRefinement] = None
    faults: Optional[FaultRefinement] = None
