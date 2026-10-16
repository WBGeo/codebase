"""
Pre-checks shared by the meshing components.
"""

from core.object_components import StructuralModelResults
from core.structural_modeling_components.structural_objects.grids.grid_classes import OctreeGrid


def pre_check_regular_grid(geomodel_result: StructuralModelResults) -> None:
    """
    Pre-check: meshing needs a structural model computed on a RegularGrid.

    The meshing components read the regular grid's layout (resolution, grid
    coordinates, a lithology block per voxel, marching-cubes surfaces whose
    vertices lie on the grid lines), which a model on an OctreeGrid doesn't
    have. Raises to block the connection rather than fail somewhere inside the
    mesher or, for the structured mesh, silently build an oversized mesh.
    """
    if isinstance(geomodel_result.structural_frame.grid, OctreeGrid):
        raise ValueError(
            "Meshing does not support structural models computed on an OctreeGrid yet. "
            "Compute the structural model on a RegularGrid for meshing."
        )
