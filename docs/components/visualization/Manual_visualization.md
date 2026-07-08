# Visualization

Visualization is a core component of the Workbench, allowing for the inspection of geological models, simulation results, and input data in both 2D and 3D. The Workbench provides built-in plotting tools and a seamless integration with **LiquidEarth** for immersive visualization.

---

## Concepts

The Workbench uses a variety of visualization methods depending on the data type and the stage of the workflow:

- **2D Slice Views:** Cross-sections through volumetric data (scalar fields, lithology blocks, simulation variables).
- **3D Interactive Plots:** Powered by [PyVista](https://docs.pyvista.org/), these allow for interactive exploration of surface meshes, point data, and volumetric grids.
- **Detailed Reports:** Text-based summaries of model results and input data.
- **Immersive Visualization (LiquidEarth):** A cloud-based platform for visualizing and sharing 3D geological models in XR (Extended Reality), VR (Virtual Reality), and AR (Augmented Reality).

---

## Workflow

Visualization is typically accessed via **inspectors** attached to Workbench components. When a component that produces a visualizable output is selected, the available visualization options appear in the inspector panel.

### 1. Visualizing Structural Models

> **Workbench components:** `Compute Structural Model`, `Compute Fault Model`

Once a structural or fault model has been computed, it can be inspected using:

- **Plot Model Result 2D:** Generates 2D cross-sections of the scalar fields and lithology.
- **Plot Model Result 3D:** Opens an interactive 3D window showing the interpolated surfaces and/or the volumetric grid.
- **Detailed Report:** Provides a summary of the stratigraphic groups, interpolation parameters, and data statistics.

### 2. Visualizing Simulation Results

> **Workbench component:** `Simulating with Sfepy`

Simulation outputs (such as pressure, temperature, or displacement fields) can be visualized using:

- **Plot Variable p at a time:** 3D visualization of a specific simulation variable at a selected time step.
- **Plot Variable T along a cross_section:** 2D slice through the simulation domain.
- **Plot Variable p along a line:** 1D profile of a variable between two points.

### 3. Visualizing Meshes

> **Workbench component:** `Plot Mesh in 3D`

Unstructured and structured meshes used for simulations can be visualized in 3D, with elements colored by their lithology or property labels.

---

## LiquidEarth Integration

LiquidEarth allows you to push your geological models to a cloud-based immersive environment.

### Setup

To use the LiquidEarth integration, you need an API token:

> **License Required**  
    Currently, a **basic license** is required to generate API tokens. Please contact [info@terranigma-solutions.com](mailto:info@terranigma-solutions.com) so a license can be issued to you.

1. Create an account at [LiquidEarth](https://liquidearth-portal.azurewebsites.net/).
2. Generate an API token in your user profile settings.

### Pushing Data to LiquidEarth

> **Workbench component:** `Push Geosolution to LiquidEarth`

1. Connect the `structural_model` output of a `Compute Structural Model` component to the input of the `Push Geosolution to LiquidEarth` component.
2. Provide your **API Token**.
3. Specify a **Space Name** (where the model will be stored) and a **Model Name**.
4. Run the component.

The component returns a **Space Link**.

### Opening in LiquidEarth

> **Workbench component:** `Open in LiquidEarth`

The output of the LiquidEarth Workbench component will display a clickable link that opens the model directly in the LiquidEarth Application.
If the Application is not installed it will link you to the installation page instead. 

---
