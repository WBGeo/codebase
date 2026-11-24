# Components

## Components
A component is the building block of the workbench.
They represent functions, turning inputs into outputs.
(In the UI, they are represented by the rounded corner blocks.
Square corners are used to show the results of a computation.)

```python
@wbgeo_component(identifier='wbgeo::creation_create_random_number',
                 title='Create Random Number',
                 description='Create a random number', color='#03b1fc')
def create_random_number(start: int, end: int = 100) -> int:
   """
   Create a new random number within bounds
   :param start: int the lower bound
   :param end: int the upper bound (defaults to 100)
   :return: random number within the given bounds
   """
   return random.randint(start, end)
```

In this example, the function `create_random_number` is declared as a component.
This is done via the `@wbgeo_component` decorator.
Both of its parameters are declared as inputs.
The second, `end`, parameter is optional (with a default value of 100).
The function's signature MUST be explicit, i.e., contain a returned type and
the type-hints of parameters.

![createRandomNumber.png](../img/createRandomNumber.png)

The picture shows the visualization of this component.

| Parameter        | Required   | Description                                                                                         |
|------------------|------------|-----------------------------------------------------------------------------------------------------|
| identifier       | required   | A unique identifier. See below for a naming scheme                                                  |
| title            | required   | The human-readable title shown to users of the workbench                                            |
| description      | (optional) | The description shown to users of the workbench on request (if absent, the pydoc is used)           |
| input_checks     | (optional) | The pre-checks (see below)                                                                          |
| color            | (optional) | The color of the component                                                                          |
| border_color     | (optional) | The border color of the component                                                                   |
| group            | (optional) | The group this component belongs to                                                                 |
| positionInGroup  | (optional) | The ranking of this item within the list of components of the same group                            |
| tags             | (optional) | For internal description                                                                            |
| return_name      | (optional) | The name of the output port(s), default "result"                                                    |
| is_object_type   | (optional) | Some components should be presented like the result (e.g. the loading component), defaults to False |
| requires_package | (optional) | The requirements which have to be installed for this component (e.g., `wbgeo[sfepy]`)               |

The execution of the component MUST NOT modify/change its inputs,
i.e. the inputs are immutable.

#### Colours, groups and order

Colours are not chosen per component. They come from one palette in
`core/workbench_palette.py`, so a workflow can be read at a glance. The colours
follow the logo: brown and green for structural modeling, teal and blue for
meshing, orange and red for process simulation. Within a step, substeps differ
in lightness or a slight shift in hue, and exports take a shade of the step
whose result they export:

| Constant             | Colour    | Used for                                                            |
|----------------------|-----------|---------------------------------------------------------------------|
| `STRUCTURAL_INPUT`   | `#d4ad6e` | Input data and grid (Regular Grid, Input data, Fault Input Data)    |
| `FAULT_MODEL`        | `#9cc65a` | Compute Fault Model                                                 |
| `STRUCTURAL_MODEL`   | `#5fa040` | Compute Structural Model                                            |
| `STRUCTURAL_EXPORT`  | `#a9a845` | Export of structural models to external tools (LiquidEarth)         |
| `MESH`               | `#3a9fc4` | Mesh creation                                                       |
| `MESH_ENGINEERING`   | `#7fd0cb` | Engineering objects (Load Wells, Sources, Shafts, Planes, Ellipses, ...) |
| `MESH_EXPORT`        | `#7896d0` | Export Mesh                                                         |
| `SIMULATION_BUILDER` | `#f9a93a` | Problem builders                                                    |
| `SIMULATION_RUN`     | `#e8663e` | Simulation runs                                                     |
| `SIMULATION_EXPORT`  | `#ec8468` | Export Simulation Results                                           |
| `PARAMETER`          | `#e6e6e6` | Parameter types: file paths, tuples, selections, option forms       |
| `BORDER`             | `#000000` | Border of all components                                            |

The same rules apply to data types (see [Types](types.md)): a type that carries
results between components takes the colour of the component that produces it
(e.g. `MeshResults` uses `MESH`), and parameter types use `PARAMETER`.

The Workbench draws component text black in light mode and white in dark mode.
Every colour has a contrast of at least 6:1 against black and about 2:1 or more
against white; avoid very pale colours, white text is unreadable on them.

```python
from core import workbench_palette as palette

@wbgeo_component(identifier='wbgeo::meshing_load_well_from_csv', title='Load Wells',
                 color=palette.MESH_ENGINEERING, border_color=palette.BORDER,
                 group='Meshing', positionInGroup=890)
```

When you add a component:

- **Reuse a colour first.** A new component of an existing kind (another mesher,
  another problem builder) uses the existing constant. Add a constant only for
  a new kind of component (a new substep).
- **New substep:** stay in the hue of the step (brown/green, teal/blue,
  orange/red) and stay clearly distinguishable from the sibling shades, also
  for colour-blind users.
- **New workflow step:** take a new hue from the logo. The yellow between blue
  and orange (around `#efb540`) is still free.
- **Contrast:** at least 4.5:1 against black (light mode) and about 2:1
  against white (dark mode).
- **No literal colours** in decorators; use the palette constants.

The sidebar lists the groups in a fixed order (Inputs, Meshing, Simulation,
Visualisation, ...) and, within a group, the components by `positionInGroup`
in **descending** order: higher numbers are listed first. Set both for every
component, ordered as the components are used in the workflow:

- each subgroup gets a block of 100, counting down: the first subgroup 990,
  980, ..., the next 890, 880, ... (e.g. in *Simulation*: builders 990–950,
  runners 890–880);
- inside a block, use steps of 10 so new components can be inserted;
- exports always sit at 0, at the bottom of the group;
- positions must be unique within a group.

`tests/test_workbench_palette.py` checks these rules (palette colours only,
group and position set, unique positions, contrast). Colours, groups and
positions are display only: changing them does not affect saved pipelines.

To be able to uniquely identify each component, you MUST follow the following naming scheme:
`identifier="wbgeo::[semantic_group]_[component_name]".`
For example, `identifier="wbgeo::interpolation_rbf", tags=["Interpolation"]`

The `is_object_type` parameter can be used to display the component like a result,
which is just loaded.
Like in the following example, the number 42 is loaded:

![load42.png](../img/load42.png)

Similarly, when using non-`controlled` inputs, the input ports are always rendered:

![create2List.png](../img/create2List.png)


#### Multi-Outputs

By defining a fixed tuple as the output of a function,
multiple outputs can be defined.
Each item of the tuple is rendered as a separate output port.
Each item's type must be a workbench compatible type, 
i.e. either a built-in, an annotated script type, or a wbgeo type.

````python
@wbgeo_component(identifier='wbgeo::create_three_things',
                 title='Create Three Things')
def create_three_things() -> typing.Tuple[int, str, int]:
   """
   :return: returns three things
   """
   return 1, "example", 2
````

#### User-Feedback

To provide feedback to a user,
any output to the standard error stream results in the job completing with a warning.

````python
import sys
print("This results in a warning, consider using a greater threshold value", file=sys.stderr)
````
Each warning should include hints to the user how to "fix" the possible problem.

Any thrown error results in the job failing with the exceptions message being used as the reason:
````python
raise ValueError("Data within the extent does not contain a good candidate for a magic unicorn. Consider using a different extend")
````
Each error should include hints to the user how to "fix" the problem.
If possible, known constraints (e.g., a parameter must fall within a range,
faults must not be present in the data, etc.) should also be checked via pre-checks (see below).
(But pre-checks can be ignored, so they should be checked within the function as well).

#### Pre-checks

Unlike errors during execution,
pre-checks indicate incompatibilities before execution.

In case a component that has been executed (and thus, has a value present)
is connected as an input to your component,
all pre-checks are run.
Pre-checks are functions referenced via the `input_checks` parameter of the components decorator.
Their parameters must be a matching subset of the inputs of the function.

In case a pre-check raises an exception, users will be warned about the problematic connection.

````python
def pre_check_my_interpolator(data: MyDataType):
    raise ValueError("Faults are not supported with the example interpolator. Consider using a different interpolator")

@wbgeo_component(input_checks=[pre_check_my_interpolator], ...)
def my_example_interpolator(data: MyDataType, treshold: int) -> MyResultType:
   # run the pre-checks first (at least those, that are really required)
   pre_check_my_interpolator(data)
   # do execution
   return MyResultType(...)
````

(Note: Only those pre-checks are run whose inputs have already been computed/are present.
This requires the invocation of the checks during the execution again.)

### Inspection components

To enable users to inspect a result,
special inspect functions can be defined.
Their textual output (`print(str)`), matplotlib rendering,
as well as pyvista plotter output is rendered on the UI.
Unlike normal components,
inspection components do not specify a return type.
(By default, they also receive the `__visualizer` tag,
hiding them from the default list of components.)

Note regarding the identifier:
The semantic group of inspectors MUST BE `inspect`,
followed by the identifier of the type,
optionally followed by a suffix denoting the specific visualization used.
Examples can be `wbgeo::inspect_my_complex_data_type` or
`wbgeo::inspect_my_complex_data_type_2d`.

They can use an optional `_inspector` helper,
which allows them to trace the computation's intermediate results.

`````python
@wbgeo_component(identifier='wbgeo::inspect_my_complex_data_type',
                 title='Visualize Example',
                 description='...')
@wbgeo_inspector()
async def _visualize_complex(i: MyComplexDataType, _inspector: InspectorHelper):
   #
   # Option 1: Use print(...) to output something
   print(i.name)
   print(":")
   print(str(i.numbers))
   print("----")

   # retrieve a value 
   # (note: this API is subject to change)
   trace = await _inspector.trace(MyListOfNumbers)
   if trace:
      print("value", await trace.get_value(), "end-value")
   else:
      print("no List of numbers found")

   # Option 2: Use the pv.Plotter
   import pyvista as pv
   mesh = pv.Cube()
   another_mesh = pv.Sphere()
   pl = pv.Plotter()
   pl.add_mesh(mesh, color='red', style='wireframe', line_width=4)
   pl.add_mesh(another_mesh, color='#03b1fc')
   pl.show()

   # Option 3: Use matplotlib
   import matplotlib.pyplot as plt
   import numpy as np

   xpoints = np.array([1, 8])
   ypoints = np.array([3, 10])

   plt.plot(xpoints, ypoints)
   plt.show()
`````

### Import / Export components
While each computation's result is stored by the workbench,
some workflows require interaction outside the workbench.

By using the `nodesapi.BasicallyABufferedFile` type as an input or as the resulting type,
users can upload/import files to a workflow or download an export of data.

The following component exports data to a `io.BytesIO` buffer,
when is then downloadable by a user:

```python
from py_api_wbgeo.nodesapi import wbgeo_component, BasicallyABufferedFile
@wbgeo_component(title='Export MyListOfNumbers',
                 description='Export MyListOfNumbers to file',
                 group='Export',
                 identifier='py::export_MyListOfNumbers_to_file',
                 )
def export_to_file(mln: MyListOfNumbers) -> BasicallyABufferedFile:
  buf = io.BytesIO() # create new (buffered in memory) file
  # write to the buffer
  buf.write('My text...'.encode('utf-8'))
  # we can specify a name for the downloaded file
  buf.filename = 'numbers.csv'
  # as long as we return a BytesIO (buffered file)
  return buf
```

The component is visible on the left side of the following picture.

![importExport.png](../img/importExport.png)

An example of an import is shown to the right and below:
Here the import is


```python
@wbgeo_component(description='Import MyListOfNumbers from F',
                 title='Import MyListOfNumbers from F',  # The title shown in the GUI
                 group='Import',
                 identifier='py::import_MyListOfNumbers',
                 return_name='mln'
                 )
def import_mesh_results_to_vtu(file: BasicallyABufferedFile) -> MyListOfNumbers:
  s = file.readlines()
  print("Loading file", file.filename)
  ret = [int(i.decode('utf-8')) for i in s[1:]]
  if len(ret) != int(s[0].decode('utf-8')):
    raise ValueError('Expected length did not match')
  return ret
```

## Dependant Inputs

One special kind of inputs are "dependant inputs",
 i.e. inputs that depend on another input.

For example: Given a list of group names and a CSV input file, the user 
 should be able to select the group of each feature.


```python
from py_api_wbgeo.smartcontrols import CtrlGroup, CtrlOrderedGrouping, SmartInputFormData, SmartInput

SmartStructuralInputSmartInputOptions = typing.Annotated[
  StructuralInputMappingOptions, SmartInput(inputs=['surface_points_file', 'group_names'],
                                          to_form=structural_input_smart_options,
                                          to_data=structural_input_smart_options_to_data)]


@wbgeo_component("...")
def structural_input_smart_options(surface_points_file: SurfaceCSVFileDataType,
                                   group_names: GroupNames
                                   ) -> CtrlGroup:
   return CtrlGroup(id='root', inner=[
      CtrlOrderedGrouping(id='groups',
                          label='Assign Groups',
                          groups=["c1", "c2"], # load this from the inputs
                          items=["a", "b", "c"], # load this from the inputs
                          )
   ]
                    )

@wbgeo_component(...)
def structural_input_smart_options_to_data(
      _input: SmartInputFormData,
      group_names: GroupNames,
) -> StructuralInputMappingOptions:
   input_as_json = smartcontrols.form_data_as_dict(_input)
   grouped = collections.defaultdict(list)
   if "root" in input_as_json and "groups" in input_as_json["root"]:
      for gn in group_names:
         if gn in input_as_json["root"]["groups"]:
            grouped[gn].extend(input_as_json["root"]["groups"][gn])

   return {k: tuple(v) for k, v in grouped.items()}


@wbgeo_component(...)
def my_component(
      surface_points_file: SurfaceCSVFileDataType = 'examples/synthetic_examples/model1/input_data/geological_data/model1_surface_points_df.csv',
      group_names: GroupNames = ['Strat_Series1'],
      mapping_object: SmartStructuralInputSmartInputOptions = {"Strat_Series1": ('rock2', 'rock1')}
) -> InputData_StructuralElements:
    pass
```
 

By using an annotated input type, with an annotation to `SmartInput`,
 an input type is defined as having "smart controls".

Parameters in the smart-input helper component functions with the same name and type 
as the real component will be passed along.

Please take a look at the smart controls file to see a list of available controls.

