"""Colour code of the Workbench components and data types.

The colours follow the logo: brown and green layers for structural modeling,
a teal/blue mesh for meshing, an orange/red grid for process simulation.
Within a step, substeps differ in lightness or a slight shift in hue, so they
stay distinguishable for colour-blind users as well.

The Workbench draws component text black in light mode and white in dark mode.
Every fill has a contrast of at least 6:1 against black (WCAG AA needs 4.5:1)
and about 2:1 or more against white; very pale fills are avoided because
white text becomes unreadable on them.

Rules:

- A component takes the colour of its step and substep.
- Exports and external tools take a shade of the step whose result they
  export (``STRUCTURAL_EXPORT``, ``MESH_EXPORT``, ``SIMULATION_EXPORT``).
- A data type that carries results between components (ports and result
  objects) takes the colour of the component that produces it.
- Parameter types (file paths, tuples, selections, option forms) share
  ``PARAMETER``, so step colours always mean "data from this step".

Adding colours:

- Reuse first. A new component of an existing kind (another mesher, another
  problem builder) uses the existing constant. Add a constant only for a new
  kind of component (a new substep).
- A new substep stays in the hue of its step (brown/green, teal/blue,
  orange/red) and is clearly distinguishable from its siblings.
- A new workflow step gets a new hue from the logo; the yellow between blue
  and orange (around ``#efb540``) is the free one.
- Every colour needs a contrast of at least 4.5:1 against ``BORDER`` (black)
  and should keep about 2:1 against white for dark mode.
- Decorators use these constants only, never literal colours.
  ``tests/test_workbench_palette.py`` checks all of this.

Order in the sidebar: the Workbench lists groups in a fixed order and, within
a group, components by ``positionInGroup`` in DESCENDING order (higher first).
Each subgroup of a group gets a block of 100, counting down in workflow order
(first subgroup 990, 980, ...; next subgroup 890, 880, ...), with steps of 10
so components can be inserted; exports sit at 0, the bottom of the group.
Every component with a ``group`` needs a ``positionInGroup``, unique within
its group.

Colours, groups and positions are display only; identifiers stay unchanged,
so saved pipelines are not affected.
"""

# Structural modeling: light brown for input data, greens for computations
STRUCTURAL_INPUT = "#d4ad6e"
"""Input data and grid (Regular Grid, Structural Input Data, Fault Input Data)."""
FAULT_MODEL = "#9cc65a"
"""Compute Fault Model and its result."""
STRUCTURAL_MODEL = "#5fa040"
"""Compute Structural Model and its result."""
STRUCTURAL_EXPORT = "#a9a845"
"""Export and load of structural models (Export Structural Model, Load Structural Model, Push Structural Model to LiquidEarth)."""

# Meshing: light teal for engineering objects, blue for mesh creation
MESH_ENGINEERING = "#7fd0cb"
"""Engineering objects (wells, sources, shafts, planes, ellipses, triangulations)."""
MESH = "#3a9fc4"
"""Mesh creation and its result."""
MESH_EXPORT = "#7896d0"
"""Export Mesh."""

# Process simulation: orange for problem builders, red for runs
SIMULATION_BUILDER = "#f9a93a"
"""Problem builders and the problems they produce."""
SIMULATION_RUN = "#e8663e"
"""Simulation runs and their results."""
SIMULATION_EXPORT = "#ec8468"
"""Export Simulation Results."""

# Shared
PARAMETER = "#e6e6e6"
"""Parameter types: file paths, tuples, selections, option forms."""
BORDER = "#000000"
"""Border colour of all components."""
