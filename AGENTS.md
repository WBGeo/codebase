# AGENTS.md

Guidance for coding agents (and humans) working on the WBGeo backend. It
complements the [README](README.md), the [contributing guide](Contributing.md) and
the [developer docs](docs/developers/index.md); where they go into detail, this file
links to them instead of repeating them.

## What WBGeo is

WBGeo is an open-source workbench for geoscientific workflows: structural geological
modeling, mesh generation, process simulation and visualization in one environment.
It is a **component-and-connector system**. A workflow is a graph of components; each
component wraps one method or processing step behind typed input and output ports.

- **Two front doors, one code base.** The same Python functions run in plain Python
  scripts/notebooks and in the visual editor (the "Workbench", a web UI that runs
  this backend in Docker). Every change has to work in both.
- **Components are black boxes.** A component only sees its typed inputs. It must
  not reach into how an upstream result was produced. Example: meshing receives a
  `StructuralModelResults` and must not access scalar fields or interpolation
  methods or re-run interpolation; anything meshing needs (e.g. surfaces extended
  past the model extent) is computed in the structural modeling step and stored on
  the result. (The `_inspector` of inspection components is an exception.)
- **Results are self-contained.** Results deliberately carry everything later
  components might need, even redundantly, so components stay independent after
  computation. Storage size is not a goal.
- **Accessibility.** Users assemble workflows without programming. Names,
  descriptions, defaults and pre-check messages are user interface: write them for
  geoscientists, not for developers.

## Repository map

| Path | Contents |
|---|---|
| `core/` | The component library |
| `core/structural_modeling_components/` | Structural modeling (`general.py`, `general_faults.py`, `interpolator_functions/`, `structural_objects/`) |
| `core/meshing_components/` | Meshing: implicit, structured/unstructured explicit, export formats |
| `core/simulation_components/` | Process simulation (SfePy, GOLEM), result loading, plots, export |
| `core/loading_components/` | Input data components (grid, structural input data, fault input data) |
| `core/liquidEarth_components/` | LiquidEarth integration |
| `core/object_components.py` | Shared result and data types |
| `core/workbench_palette.py` | Colours of all components and types (see "Component rules") |
| `core/utility/pydantic_bridge.py` | Pydantic aliases for third-party types |
| `examples/` | Synthetic example workflows (`synthetic_examples/model1` ... `model9`) and case studies |
| `tests/` | Test suite, one subpackage per `core/` area |
| `docs/` | MkDocs source of the documentation at https://wbgeo.github.io/ |
| `docker-compose.yml`, `docker/` | Local Workbench (visual editor) stack |

## Setup and commands

- Python **3.12.4 or newer** (see `pyproject.toml`). Use a virtual environment in
  the repository root (`venv/`).
- Install: `pip install -r requirements.txt`. The file adds extra package indexes
  (PyTorch CPU wheels, and a public package registry that provides `py_api_wbgeo`,
  the decorator API). On Windows, `sfepy` needs a C compiler to build.
- Run code from the repository root, or put the root on `PYTHONPATH`; imports are
  absolute (`from core....`).
- Run an example: `python examples/synthetic_examples/model1/WBGeo1.0_model1.py`

Tests (pytest):

```bash
# limit native threads: libraries otherwise oversubscribe the CPU and some tests get 10-20x slower
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1

python -m pytest tests/path/to/test_file.py -q        # the tests for what you changed
python -m pytest tests -q -m "not slow"               # quick pass, skips integration tests
python -m pytest tests -q                             # full suite, roughly 8-15 minutes
```

- Run the tests of the code you touched after each change; run the full suite
  before opening or updating a merge request.
- Long-running end-to-end tests are marked `@pytest.mark.integration` and
  `@pytest.mark.slow`.
- `--show-plots` / `--show-plots-always` open plots from tests that support them.
- `tests/test_component_signatures.py` checks that every component in `core/` is
  correctly declared; `tests/test_workbench_palette.py` checks colours and sidebar
  order. Both are fast; run them whenever you touch a decorator.
- On Windows, deep checkout paths can exceed `MAX_PATH` in tests that read the
  example CSV files; use a short checkout or worktree path.

Docs: CI builds them with
`pip install mkdocs-material "mkdocstrings[python]" mkdocs-gen-files mkdocs-literate-nav`
and `mkdocs build` (after adding `__init__.py` to every directory under `core/`).

## Components and types

Read [docs/developers/components.md](docs/developers/components.md) and
[docs/developers/types.md](docs/developers/types.md) before adding or changing one.
The essentials:

- A **component** is a function decorated with `@wbgeo_component(...)` from
  `py_api_wbgeo.nodesapi`. Its parameters (with type hints) become input ports, its
  return type the output port.
- A **type** is a Pydantic model decorated with `@wbgeo_type(...)`, or a builtin
  annotated with `typing.Annotated[..., AnnotatedScriptType(...)]`.
- In the Workbench, every input is deserialized from JSON before a component runs
  and every result is serialized after it ran (Pydantic). To check this in a plain
  Python script, call `register_as_test_nodes_api()` from
  `examples/pydantic_nodesapi_simulation.py` before any other import.
- What the Workbench shows:
  - component name: `title=`;
  - component help text: `description=`, falling back to the docstring;
  - result object name: the `name=` of the **returned type**;
  - return port label: `return_name=`.
  - input port labels: argument names
  - (see the parameter table of [components.md](docs/developerts/components.md))

## Component rules

1. **Identifiers are permanent.** Saved pipelines refer to components and types by
   `identifier`. Never change an existing identifier. New ones follow
   `wbgeo::[semantic_group]_[component_name]`; inspectors use the semantic group
   `inspect` (`wbgeo::inspect_<type>[_suffix]`).
2. **Inputs are immutable.** A component must not modify its inputs; copy before
   changing anything.
3. **Names are user interface.**
   - Titles start with a verb and name the object: "Compute Structural Model",
     "Create Unstructured Explicit Mesh", "Load Wells" (plural when a file can
     hold several), "Build GOLEM Hydrothermal Problem".
   - Type names (= result object names) are short Title Case nouns: "Structural
     Model", "Mesh", "Simulation Results". No "Result of ...", no snake_case.
   - `return_name` is lowercase snake_case.
   - Titles, names, colours and positions are display only and can be changed;
     update the manuals in `docs/components/` when a title changes.
4. **Pre-checks** (`input_checks=[...]`) report incompatible connections before
   execution. Call them again at the start of execution (they only run in the UI
   when the inputs exist). Every pre-check needs a test.
5. **Colours and order** come from `core/workbench_palette.py`, never literal colour
   strings.
   - Each component takes the colour of its step and substep; exports take a shade
     of their step; a result type takes the colour of the component that produces
     it; parameter types use `PARAMETER`.
   - The sidebar sorts `positionInGroup` in **descending** order: blocks of 100
     counting down (990, 980, ... / 890, ...), exports at 0, unique per group.
   - The group order is fixed in the frontend: Inputs, Meshing, Simulation,
     Visualisation.
   - The rules for new colours are in the palette module docstring.
6. **No `Union[...]` in component signatures for new ports.** The visual editor
   cannot connect union-typed ports yet.
7. **Docstrings** explain the component and each input in user terms. Inspectors
   (`@wbgeo_inspector`) may `print()`: their output is rendered in the UI.
8. **Every new component** gets a fast unit test on small input and is used in an
   example workflow that runs.

## Types and results

- Types must be valid, serializable Pydantic models; add a test that a type
  survives a JSON round trip.
- **Never use `arbitrary_types_allowed`.** Use the aliases instead:
  - `numpy.ndarray`: `NpNDArrayFp64` / `NpNDArrayInt64` from `pydantic_numpy`;
  - `pandas.DataFrame`: `PandasDataFrame` from `core.utility.pydantic_bridge`;
  - `meshio.CellBlock`: `MeshIOCellBlock` from `core.utility.pydantic_bridge`.

  If a new third-party type needs support, add an alias in `pydantic_bridge.py`.
  (`MeshResults` still has the setting; it is legacy, do not copy it.)
- **Changing result classes** (`StructuralFrame`, `FaultFrame`,
  `StructuralModelResults`, `MeshResults`, `SimulationResults`, ...):
  - keep every existing attribute and its meaning;
  - prefer reusing attributes over parallel new fields, and helper methods over
    stored duplicates;
  - ask before removing anything.

## Code style

- Use `logging.getLogger(__name__)`, not `print()` (except in inspectors): `debug`
  for progress, `info` for milestones, `warning` for unexpected but non-fatal cases.
- Consistent type hints, including Pydantic fields.
- Keep changes focused. Don't reformat or rename unrelated code in the same change.
- Synthetic example scripts that call `create_structured_mesh_data` pass
  `refinement_data` explicitly; the default only fits model 2.

## Domain conventions worth knowing

- `structural_elements` are ordered **youngest to oldest**.
- Interpolators return scalar fields in **ZYX** shape `(nz, ny, nx)`; they are
  stored on the groups in **XYZ** `(nx, ny, nz)`. Use asymmetric grid resolutions
  (e.g. 5x4x3) in tests to catch axis-order bugs.
- `fault_activity[fault]` is the index of the youngest affected group; a fault is
  active for a group if `group_index >= that index`.
- Structural integration tests generate their input analytically inline and
  compare against an analytical expected lithology.
- SfePy cell data uses `mat_id` (0-based, mesh block index); GOLEM results use
  `ObjectId` (1-based Exodus block id). `SimulationResults.block_names` maps both
  to lithology names.

## Running the Workbench locally (Docker)

- `docker compose up -d`, then open http://localhost:8080.
- **Run your local code** by uncommenting the "option 1" volume mounts of the
  `py_runner` service in `docker-compose.yml` (`./core/`, `./requirements.txt`,
  `./examples/`) and commenting out the `own_data` line. This is a local setting:
  **never commit it**.
- After changing decorators (titles, colours, positions, types), **restart
  `backend` first, then `py_runner`**. The runner only creates definitions and
  never updates existing ones, and the backend keeps them in memory.
- If `py_runner` fails after an image update with
  `BaseModel.__init__() takes 1 positional argument`, its package cache is stale:

  ```bash
  docker compose rm -sf py_runner
  docker volume rm codebase_pythonDepCache
  docker compose up -d py_runner
  ```

## Git, issues and merge requests

- The public home of the project is https://github.com/WBGeo/codebase; the
  **issues** live there. The team currently still develops on an internal GitLab
  instance and mirrors `main` to GitHub; the full move is tracked in
  https://github.com/WBGeo/codebase/issues/10.
- Work on a branch, open a merge (pull) request against `main`, and run the full
  test suite first.
- **Reference issues by full URL** in commits and merge requests, e.g.
  `Closes https://github.com/WBGeo/codebase/issues/4`. While development still
  runs through the internal instance, a bare `#4` would point to the wrong issue
  tracker; the URL form works everywhere and closes the GitHub issue once the
  commit reaches `main`.
- Don't commit local files: the Docker dev mounts, `own_data/`, caches such as
  `examples/case_studies/*/input_data/topography_cache/`, or credentials (a local
  `.env` holds tokens, e.g. for LiquidEarth).
- Docs and docstrings link to https://wbgeo.github.io/ and to github.com URLs,
  never to the internal GitLab instance (it is not publicly accessible).
