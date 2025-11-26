# Codebase

This document outlines the functionality of the internal,
decorator-based DSL for component definitions.

The DSL is available as a decorator-based API via pip at
`pip install py_api_wbgeo`
and the decorators can be imported using:

`````python
from py_api_wbgeo.nodesapi import *
`````

### Components

A component is a building clock

```python
@wbgeo_component(identifier='py::create_random_number',
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

 Parameter      | Required   | Description                                                
----------------|------------|------------------------------------------------------------
 identifier     | required   | A unique identifier. See below for a naming scheme         
 title          | required   | The human-readable title shown to users of the workbench   
 description    | required   | The description shown to users of the workbench on request 
 input_checks   | (optional) | TODO                                                       
 color          | (optional) | The color of the component                                 
 border_color   | (optional) | the border color of the component                          
 group          | (optional) | TODO                                                       
 return_name    | (optional) | The name of the output port, default "result"              
 is_object_type | (optional) | TODO                                                       

The execution of the component MUST NOT modify/change its inputs,
i.e. the inputs are immutable.

**TODO**: identifier rule

#### User-Feedback

**TODO**: STDERR/exceptions

#### Pre-checks

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

| parameter  | required | description |
|------------|----------|-------------|
| identifier | required | todo        |
| todo       | todo     | todo        |

#### Defining more complex types

For more complex types,

````python
@wbgeo_type(name='Input data for a geological model', color='orange',
            identifier='MyComplexDataType')
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

They can use an optional `_inspector` helper,
which allows them to trace the computation's intermediate results.

`````python
@wbgeo_component(identifier='py::__visualize_complex',
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
