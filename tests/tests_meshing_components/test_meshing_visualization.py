"""Plot Mesh in 3D (meshing_visualization.py): style dropdown and inspector."""
import asyncio

from core.meshing_components.meshing_visualization import meshing_visualization as mv
from core.object_components import StructuralModelResults


def test_mesh_style_is_a_select_of_the_three_styles():
    assert mv.MeshStyleType.__metadata__[0].kwargs["controlled"] == "Select|surface|wireframe|points"


def test_mesh_inspector_uses_the_traced_structural_model_and_the_style(monkeypatch, fake_inspector):
    calls = []
    monkeypatch.setattr(mv, "plot_mesh_3d", lambda mesh, geo, style, show: calls.append((mesh, geo, style)))
    inspector = fake_inspector({StructuralModelResults: "geo"})

    asyncio.run(mv.inspect_implicit_mesh_3d("mesh", inspector))
    asyncio.run(mv.inspect_implicit_mesh_3d("mesh", inspector, style="wireframe"))

    assert calls == [("mesh", "geo", "surface"), ("mesh", "geo", "wireframe")]
