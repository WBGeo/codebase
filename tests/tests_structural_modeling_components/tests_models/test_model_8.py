from __future__ import annotations

import numpy as np
import pytest

# If UCK depends on GemPy, skip cleanly when not installed:
gempy = pytest.importorskip("gempy")  # remove if UCK in your project doesn't require it


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.parametrize("method", ["RBF", "UCK"])
def test_synthetic_model_runs_and_produces_reasonable_outputs(method):
    """
    Integration test: run the full synthetic model at the required resolution,
    using either RBF (no faults or faults ignored) or UCK (fault domains).
    """
    # --- Build the same synthetic inputs every time ---
    # TODO: replace these two lines with your real builder(s)
    # frame, fault_frame = build_synthetic_model(required_resolution=True)
    # or: input_data = make_synthetic_input(); frame = build_structural_frame(input_data); fault_frame = build_fault_frame(...)
    frame, fault_frame = build_synthetic_model_for_tests()  # <-- you provide

    # --- Configure interpolators ---
    for g in frame.structural_groups:
        # IMPORTANT: your groups require context first
        sp = frame.surface_points
        pts = sp[["X", "Y", "Z"]].to_numpy()
        g.update_interpolation_context(pts)

        if method == "RBF":
            g.set_interpolation_method("Radial Basis Function")
        else:
            # UCK here likely means: faults computed via GemPy UCK,
            # while groups can still be RBF (or your choice).
            # If you mean UCK for groups too, switch this accordingly.
            g.set_interpolation_method("Radial Basis Function")

    # --- Run the pipeline ---
    # TODO: replace with your real entry point
    # result = compute_structural_model(frame, fault_frame=fault_frame, extract_meshes=True, ...)
    result = run_full_pipeline(frame=frame, fault_frame=fault_frame, method=method)  # <-- you provide

    # --- Stable assertions (avoid exact numeric comparisons) ---

    # 1) scalar fields exist and have expected shape
    grid = frame.grid
    assert grid is not None
    for g in frame.structural_groups:
        sf = g.get_scalar_field()
        assert sf.shape == tuple(grid.resolution)
        assert np.isfinite(sf).all()
        assert float(np.nanstd(sf)) > 0.0  # not constant

        # scalar values assigned to elements
        for e in g.structural_elements:
            assert e.scalar_value is not None
            assert np.isfinite(e.scalar_value)

    # 2) lithology block exists (if your pipeline produces one)
    # If result exposes it, assert shape/dtype-ish properties
    if hasattr(result, "lithology_block") and result.lithology_block is not None:
        lith = result.lithology_block
        assert lith.shape == tuple(grid.resolution)
        # should be integer-like
        assert np.issubdtype(lith.dtype, np.integer)

    # 3) Faults / UCK-specific checks
    if method == "UCK":
        assert fault_frame is not None
        assert fault_frame.domain_map is not None
        dm = fault_frame.domain_map
        assert dm.shape == tuple(grid.resolution)
        assert np.isfinite(dm).all()
        # at least 1 domain; often >1 if faults actually split the volume
        assert len(np.unique(dm)) >= 1

        # each fault should have scalar_field + scalar_value
        for f in fault_frame.fault_elements:
            assert f.scalar_field is not None
            assert f.scalar_value is not None
            assert f.scalar_field.shape == tuple(grid.resolution)


@pytest.mark.integration
@pytest.mark.slow
def test_fault_domain_plotting_does_not_crash():
    """
    Optional: "smoke test" plotting (no image comparison).
    Only checks: doesn't throw.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless backend for CI

    # TODO: build the same model
    frame, fault_frame = build_synthetic_model_for_tests()
    assert fault_frame is not None

    # If plotting requires domains computed first, do that here (or reuse pipeline):
    # run_full_pipeline(...)

    fault_frame.plot_fault_domain_section(axis="y", index=12)
    plot_fault_model_3D(fault_frame)  # your helper
