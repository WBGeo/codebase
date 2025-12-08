# Codebase

This document outlines the functionality of the internal,
decorator-based DSL for component definitions.

The DSL is available as a decorator-based API via pip at
`pip install py_api_wbgeo`

The decorators can be imported using:

`````python
from py_api_wbgeo.nodesapi import *
`````

Currently, the packages is deployed
to a private [GitLab registry](https://git.rwth-aachen.de/wbgeo/proof-of-concept-backend/-/packages).

* Create
  a [personal access token (PAT)](https://git.rwth-aachen.de/-/user_settings/personal_access_tokens)
  with `read_api` permissions (`read_registry` is not sufficient) on the
  git.rwth-aachen.de instance
* Install the package via
  `pip install py_api_wbgeo --index-url https://gitlab-ci-token:<your_personal_token>@git.rwth-aachen.de/api/v4/projects/102532/packages/pypi/simple`

As an alternative: download the latest package as a wheel-file
from [GitLab's package registry](https://git.rwth-aachen.de/wbgeo/proof-of-concept-backend/-/packages)
and install it manually via `pip install py_api_wbgeo...whl`

### Components

A component is the building block of the workbench.
They represent functions, turning inputs into one output.
(In the UI, they are represented by the rounded corner blocks.
Square corners are used to show the results of a computation.)

```python
@wbgeo_component(identifier='wbgeo::creation_create_random_number',
                 title='Create Random Number',
                 description='Create a random number', color='#03b1fc')
def create_random_number(start: int, end: int = 100) -> int:
   return random.randint(start, end)
```

In this example, the function `create_random_number` is declared as a component.
This id done via the `@wbgeo_component` decorator.
Both of its parameters are declared as inputs.
The second, `end`, parameter is optional (with a default value of 100).
The function's signature MUST be explicit, i.e., contain a returned type and
the type-hints of parameters.

| Parameter      | Required   | Description                                                                                         |
|----------------|------------|-----------------------------------------------------------------------------------------------------|
| identifier     | required   | A unique identifier. See below for a naming scheme                                                  |
| title          | required   | The human-readable title shown to users of the workbench                                            |
| description    | (optional) | The description shown to users of the workbench on request (if absent, a pydoc must be present)     |
| input_checks   | (optional) | The pre-checks (see below)                                                                          |
| color          | (optional) | The color of the component                                                                          |
| border_color   | (optional) | The border color of the component                                                                   |
| group          | (optional) | TODO                                                                                                |
| tags           | (optional) | TODO                                                                                                |
| return_name    | (optional) | The name of the output port, default "result"                                                       |
| is_object_type | (optional) | Some components should be presented like the result (e.g. the loading component), defaults to False |

The execution of the component MUST NOT modify/change its inputs,
i.e. the inputs are immutable.

To be able to uniquely identify each component, you MUST follow the folling naming scheme:
`identifier="wbgeo::[semantic_group]_[component_name]".`
For example, `identifier="wbgeo::interpolation_rbf", tags=["Interpolation"]`

**TODO**: identifier rule

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

> ! the following

### Types

The workbench, by default, supports various built-in types.
In the following table, the existing

| Python-Type | Description                                                                                             |
|-------------|---------------------------------------------------------------------------------------------------------|
| `bool`      | A boolean value, represented as a checkbox in the UI                                                    
| `int`       | A numeric, integer value, represented as a number field in the UI                                       
| `float`     | A floating point numeric value, represented as an number field allowing floating point values in the UI 
| `str`       | A text value, represented as a text field in the UI                                                     
| `List[?]`   | A list of values, without an input representation in the UI                                             

TODO: file input ("upload")


#### Annotating existing types

To help users by giving a type a semantic meaning,
we can annotate types using `AnnotatedScriptType`, like in the following exmaple

````python
MyListOfNumbers = typing.Annotated[
   typing.List[int], AnnotatedScriptType(name='numbers', color='aqua',
                                         identifier='MyListOfNumbers')]
````

The UI will handle them like their own type (due to the unique identifier),
yet they are handled like their original type during the execution.
For example, a special type for file paths could be added.

TOOD: identifier

The `AnnotatedScriptType` accepts the following parameters:

| parameter  | required   | description                                   |
|------------|------------|-----------------------------------------------|
| identifier | required   | todo                                          |
| name       | required   | The human-readable name of this type          |
| color      | (optional) | The color of this type                        |
| controlled | (optional) | If present, a value can be entered via the UI |


The UI additionally supports special input controls via the `controlled` parameter.
By default, a parameters default value is used.

| Controlled=        | Description                             |
|--------------------|-----------------------------------------|
| text               | A generic text input                    |
| number             | A numeric text input                    |
| boolean            | A boolean input (                       |
| password           | A generic text input with masked inputs |
| Table\|C1\|...\|Cn | NYI                                     |
| file               | NYI                                     |

To add support for additional input types, 
 they have to be added to the UI (feel free to ask Alex for this).


#### Defining more complex types

For more complex types,

````python
@wbgeo_type(name='Input data for a geological model', color='orange',
            identifier='wbgeo::my_complex_data_type')
@dataclass
class MyComplexDataType:
   name: str
   numbers: MyListOfNumbers

````

TODO: controls

TODO: Refer to other `wbgeo_type` objects as fields

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
