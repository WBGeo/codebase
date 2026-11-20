"""Colour code of the Workbench components and data types.

The colours follow the logo: brown and green layers for structural modeling,
a teal/blue mesh for meshing, an orange/red grid for process simulation.
Within a step, substeps differ in lightness, so they stay distinguishable
for colour-blind users as well. Every fill has a contrast ratio of at least
6:1 against the black component text and border (WCAG AA needs 4.5:1).

Rules:

- A component takes the colour of its step and substep.
- A data type that carries results between components (ports and result
  objects) takes the colour of the component that produces it.
- Parameter types (file paths, tuples, selections, option forms) share
  ``PARAMETER``, so step colours always mean "data from this step".
- Exports and external tools share ``EXPORT``.

Adding colours:

- Reuse first. A new component of an existing kind (another mesher, another
  problem builder) uses the existing constant. Add a constant only for a new
  kind of component (a new substep).
- A new substep stays in the hue of its step (brown/green, teal/blue,
  orange/red) and differs from its siblings mainly in lightness.
- A new workflow step gets a new hue from the logo; the yellow between blue
  and orange (around ``#efb540``) is the free one.
- Every colour needs a contrast of at least 4.5:1 against ``BORDER``.
- Decorators use these constants only, never literal colours.
  ``tests/test_workbench_palette.py`` checks all of this.

Order in the sidebar (``positionInGroup``, ascending): each subgroup of a group
gets a block of 100 (first subgroup 10-99, next 110-199, ...), with steps of 10
inside a block so components can be inserted; exports sit at 900. Every
component with a ``group`` needs a ``positionInGroup``, unique within its group.

Colours, groups and positions are display only; identifiers stay unchanged,
so saved pipelines are not affected.
"""

# Structural modeling: light brown for input data, greens for computations
STRUCTURAL_INPUT = "#e8c992"
"""Input data and grid (Regular Grid, Input data for Structural Elements, Fault Input Data)."""
FAULT_MODEL = "#9cc65a"
"""Compute Fault Model and its result."""
STRUCTURAL_MODEL = "#5fa040"
"""Compute Structural Model and its result."""

# Meshing: light teal for engineering objects, blue for mesh creation
MESH_ENGINEERING = "#7fd0cb"
"""Engineering objects (wells, sources, shafts, planes, ellipses, triangulations)."""
MESH = "#3a9fc4"
"""Mesh creation and its result."""

# Process simulation: orange for problem builders, red for runs
SIMULATION_BUILDER = "#f9a93a"
"""Problem builders and the problems they produce."""
SIMULATION_RUN = "#e8663e"
"""Simulation runs and their results."""

# Shared
EXPORT = "#a7b4c2"
"""Exports and external tools (Export Mesh, Export Simulation Results, LiquidEarth)."""
PARAMETER = "#e6e6e6"
"""Parameter types: file paths, tuples, selections, option forms."""
BORDER = "#000000"
"""Border colour of all components."""
