<h1 align="center">
  <img src="docs/img/logo.png" alt="WBGeo logo" width="480">
</h1>

<p align="center"><strong>Backend code for WBGeo</strong> — a workbench for geoscientific workflows</p>

<p align="center">
  <a href="https://wbgeo.github.io/"><img alt="Documentation" src="https://img.shields.io/badge/docs-wbgeo.github.io-1f6feb"></a>
  <a href="LICENSE"><img alt="License: EUPL-1.2" src="https://img.shields.io/badge/license-EUPL--1.2-2ea043"></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776ab">
</p>

This repository contains the **backend code for [WBGeo](https://wbgeo.github.io/)**, an open-source
workbench for geoscientific workflows covering structural geological modeling, mesh generation,
process simulation, and visualization in one integrated environment. WBGeo is built around a
**component-and-connector system**: workflows are assembled from typed, interchangeable
components, each wrapping a method or processing step behind a standardized interface. The
workbench can be used directly through this Python backend, or through its visual interface
(visual DSL), which runs this same code underneath.

**📖 Full documentation, installation guide, and component manuals: [wbgeo.github.io](https://wbgeo.github.io/)**

---

## Repository structure

| Path | Contents |
|---|---|
| [`core/`](core) | The component library — see the breakdown below |
| [`examples/`](examples) | Synthetic example workflows (`synthetic_examples/model1` … `model9`) and real-world case studies (`case_studies/`) |
| [`tests/`](tests) | Test suite, one subpackage per `core/` component |
| [`docs/`](docs) | Documentation source, built with MkDocs and published at [wbgeo.github.io](https://wbgeo.github.io/) |

### `core/`

WBGeo's four modeling steps, plus the shared infrastructure they're built on:

| Component | Description |
|---|---|
| [`structural_modeling_components/`](core/structural_modeling_components) | Reconstruct 3D geological structure — stratigraphy, unconformities, faults — from surface points and orientations, using implicit interpolation methods (RBF, Kriging, and others). |
| [`meshing_components/`](core/meshing_components) | Generate 3D watertight meshes (structured, unstructured, or implicit) from a structural model, and export them to common mesh formats (VTK, Exodus, Abaqus, ANSYS, FEFLOW, GMSH, STL, RESQML, ...). |
| [`simulation_components/`](core/simulation_components) | Interface with numerical solvers — currently [SfePy](https://sfepy.org/) — using meshes produced by the previous step. |
| [`liquidEarth_components/`](core/liquidEarth_components) | Push results into [LiquidEarth](https://www.terranigma-solutions.com/liquidearth) for immersive XR/VR/AR visualization. |
| [`loading_components/`](core/loading_components) | Load geological input data (surface points, orientations, faults) and grid definitions from CSV/JSON files into the pipeline. |
| [`object_components.py`](core/object_components.py) | Shared typed data objects passed between components (`StructuralModelResults`, `MeshResults`, `SimulationResults`, ...). |
| [`utility/`](core/utility) | Shared Pydantic type adapters (e.g. for `pandas.DataFrame`, `meshio.CellBlock`) used throughout `core/`. |

Every component is registered with `@wbgeo_component`/`@wbgeo_type` (from `py_api_wbgeo`), which is
what makes it independently callable both from plain Python and from the visual interface — see
the [developer's guide](https://wbgeo.github.io/developers/) for how to add new ones.

---

## Funding

This project was funded by the Federal Ministry of Research, Technology and Space (BMFTR) under
the [Geoforschung für Nachhaltigkeit (GEO:N)](https://www.ptj.de/projektfoerderung/geo-n)
initiative (Grant no. 03G0922A).

## License

WBGeo is released under the [European Union Public Licence v1.2 (EUPL-1.2)](LICENSE).
