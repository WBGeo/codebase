
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

You must not use pydantics 'arbitrary_types' config option.
Take a look at the pydantic_bridge adapters instead.

TODO: controls

TODO: Refer to other `wbgeo_type` objects as fields
