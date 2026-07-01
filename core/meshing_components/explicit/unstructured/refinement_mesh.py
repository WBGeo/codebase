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
# Triangulation
# ------------------------
@wbgeo_type(name="Triangulation Refinement", identifier="TriangulationRefinement")
class TriangulationRefinement(BaseModel):
    """
    Mesh refinement around triangulated surfaces.
    """
    hmin: float = 3.0
    hmax: float = 30.0
    d1: float = 50.0
    d2: float = 100.0
    enabled: bool = True


# ------------------------
# Ellipses
# ------------------------
@wbgeo_type(name="Ellipse Refinement", identifier="EllipseRefinement")
class EllipseRefinement(BaseModel):
    """
    Mesh refinement around elliptical regions.
    """
    hmin: float = 3.0
    hmax: float = 30.0
    d1: float = 30.0
    d2: float = 80.0
    enabled: bool = True


# ------------------------
# Faults
# ------------------------
@wbgeo_type(name="Fault Refinement", identifier="FaultRefinement")
class FaultRefinement(BaseModel):
    """
    Mesh refinement around fault surfaces.
    """
    hmin: float = 3.0
    hmax: float = 30.0
    d1: float = 20.0
    d2: float = 60.0
    enabled: bool = True


# ------------------------
# MAIN CONTAINER
# ------------------------
@wbgeo_type(name="Mesh Refinement", identifier="Refinement")
class Refinement(BaseModel):
    """
    Container for all mesh refinement options.
    """

    wells: Optional[
        Union[LinearWellRefinement, FunctionWellRefinement]
    ] = None

    sources: Optional[
        Union[LinearSourceRefinement, FunctionSourceRefinement]
    ] = None

    triangulation: Optional[TriangulationRefinement] = None
    ellipses: Optional[EllipseRefinement] = None
    faults: Optional[FaultRefinement] = None
