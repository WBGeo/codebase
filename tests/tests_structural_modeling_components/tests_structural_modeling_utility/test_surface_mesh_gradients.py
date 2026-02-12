import numpy as np
import pytest

from core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients import (
    get_surface_mesh_gradients)


# ----------------------------
# Minimal dummy framework types
# ----------------------------

class DummyGrid:
    def __init__(self, resolution):
        self.resolution = resolution
        nx, ny, nz = resolution

        # Coordinate axes (aligned with array indices)
        xs = np.arange(nx, dtype=float)
        ys = np.arange(ny, dtype=float)
        zs = np.arange(nz, dtype=float)

        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        self._X, self._Y, self._Z = X, Y, Z

        # Flattened coordinates in the same order that .reshape(resolution) expects
        self.grid_coordinates = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)


class DummyElement:
    def __init__(self, name, points):
        self.name = name
        self._points = np.asarray(points, dtype=float)
        self.last_mesh_type = None

    def get_mesh(self, mesh_type):
        # record for test assertions
        self.last_mesh_type = mesh_type
        # Your function expects [0] to be points; return a tuple/list where first is points.
        return (self._points,)


class DummyGroup:
    def __init__(self, scalar_field, structural_elements):
        self.scalar_field = scalar_field
        self.structural_elements = structural_elements


class DummyFaultElement(DummyElement):
    def __init__(self, name, points, scalar_field):
        super().__init__(name, points)
        self.scalar_field = scalar_field


class DummyFaultFrame:
    def __init__(self, fault_elements):
        self.fault_elements = fault_elements


class DummyStructuralFrame:
    def __init__(self, grid, structural_groups, fault_frame=None):
        self.grid = grid
        self.structural_groups = structural_groups
        self.fault_frame = fault_frame


class DummyResult:
    def __init__(self, structural_frame):
        self.structural_frame = structural_frame


# ----------------------------
# Helpers / fixtures
# ----------------------------

@pytest.fixture
def base_result_no_faults():
    # Small grid
    grid = DummyGrid(resolution=(5, 6, 7))
    X, Y, Z = grid._X, grid._Y, grid._Z

    # Scalar field with known constant gradient in index-space:
    # sf = 1*X + 2*Y + 3*Z  => np.gradient(sf) should yield (1,2,3) everywhere.
    sf = 1.0 * X + 2.0 * Y + 3.0 * Z

    # A couple of elements with mesh points inside the grid
    e1 = DummyElement("LayerA", points=[[1, 1, 1], [2, 3, 4], [4, 5, 6]])
    e2 = DummyElement("LayerB", points=[[0, 0, 0], [3, 2, 1]])

    group = DummyGroup(scalar_field=sf, structural_elements=[e1, e2])
    frame = DummyStructuralFrame(grid=grid, structural_groups=[group], fault_frame=None)
    return DummyResult(frame)


@pytest.fixture
def base_result_with_faults(base_result_no_faults):
    grid = base_result_no_faults.structural_frame.grid
    X, Y, Z = grid._X, grid._Y, grid._Z

    # Fault scalar field can be different
    sf_fault = 4.0 * X + 0.0 * Y + 0.5 * Z  # gradient (4,0,0.5)

    f1 = DummyFaultElement("Fault1", points=[[1, 2, 3], [4, 5, 6]], scalar_field=sf_fault)
    fault_frame = DummyFaultFrame([f1])
    base_result_no_faults.structural_frame.fault_frame = fault_frame
    return base_result_no_faults


# ----------------------------
# Tests
# ----------------------------

def test_returns_two_dicts_and_empty_faults_when_none(base_result_no_faults, monkeypatch):
    # monkeypatch normalize_vectors to avoid depending on your real implementation
    # (and to ensure deterministic behavior)
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    grads, fault_grads = get_surface_mesh_gradients(base_result_no_faults, norm=False)

    assert isinstance(grads, dict)
    assert isinstance(fault_grads, dict)
    assert fault_grads == {}  # empty dict when no fault_frame


def test_structural_element_keys_and_shapes(base_result_no_faults, monkeypatch):
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    grads, fault_grads = get_surface_mesh_gradients(base_result_no_faults, norm=False)

    assert set(grads.keys()) == {"LayerA", "LayerB"}
    assert fault_grads == {}

    for name, payload in grads.items():
        assert "points" in payload and "vectors" in payload
        pts = payload["points"]
        vecs = payload["vectors"]
        assert pts.shape[1] == 3
        assert vecs.shape == pts.shape


def test_interpolation_matches_known_constant_gradient(base_result_no_faults, monkeypatch):
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    grads, _ = get_surface_mesh_gradients(base_result_no_faults, norm=False)

    expected = np.array([1.0, 2.0, 3.0], dtype=float)
    for payload in grads.values():
        vecs = payload["vectors"]
        # constant gradient everywhere (within floating tolerance)
        assert np.allclose(vecs, expected, atol=1e-10)


def test_normalization_on_makes_unit_vectors(base_result_no_faults, monkeypatch):
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    grads, _ = get_surface_mesh_gradients(base_result_no_faults, norm=True)

    for payload in grads.values():
        vecs = payload["vectors"]
        norms = np.linalg.norm(vecs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-10)


def test_mesh_type_is_forwarded_to_get_mesh(base_result_no_faults, monkeypatch):
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    mesh_type = "combined"
    grads, _ = get_surface_mesh_gradients(base_result_no_faults, norm=False, mesh_type=mesh_type)

    # Ensure each element recorded the mesh_type used
    group = base_result_no_faults.structural_frame.structural_groups[0]
    for elem in group.structural_elements:
        assert elem.last_mesh_type == mesh_type


def test_faults_present_when_fault_frame_exists(base_result_with_faults, monkeypatch):
    import core.structural_modeling_components.structural_modeling_utility.surface_mesh_gradients as mod
    monkeypatch.setattr(mod, "normalize_vectors",
                        lambda v: v / np.linalg.norm(v, axis=1, keepdims=True))

    grads, fault_grads = get_surface_mesh_gradients(base_result_with_faults, norm=False)

    assert "Fault1" in fault_grads
    pts = fault_grads["Fault1"]["points"]
    vecs = fault_grads["Fault1"]["vectors"]
    assert pts.shape == vecs.shape
    assert pts.shape[1] == 3

    # Fault scalar field: 4*X + 0*Y + 0.5*Z => gradient approx (4,0,0.5)
    expected_fault = np.array([4.0, 0.0, 0.5], dtype=float)
    assert np.allclose(vecs, expected_fault, atol=1e-10)
