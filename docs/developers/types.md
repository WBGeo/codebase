# Types

The workbench, by default, supports various built-in types.
In the following table, the existing and supported python types are presented.

| Python-Type   | Description                                                                                             |
|---------------|---------------------------------------------------------------------------------------------------------|
| `bool`        | A boolean value, represented as a checkbox in the UI                                                    
| `int`         | A numeric, integer value, represented as a number field in the UI                                       
| `float`       | A floating point numeric value, represented as an number field allowing floating point values in the UI 
| `str`         | A text value, represented as a text field in the UI                                                     
| `List[?]`     | A list of values, without an input representation in the UI                                             
| `Optional[?]` | An optional value, requires a default (e.g., `Optional[str] = None`)                                    

#### Annotating existing types

To help users by giving a type a semantic meaning,
we can annotate types using `AnnotatedScriptType`, like in the following example:

````python
from py_api_wbgeo.nodesapi import AnnotatedScriptType
MyListOfNumbers = typing.Annotated[
   typing.List[int], AnnotatedScriptType(name='numbers', color='aqua',
                                         identifier='MyListOfNumbers')]
````

The UI will handle them like their own type (due to the unique identifier),
yet they are handled like their original type during the execution.
For example, a special type for file paths could be added.

The identifier must be unique.
(We recommend a namespaced identifier to avoid naming collisions.)

The `AnnotatedScriptType` accepts the following parameters:

| parameter  | required   | description                                  |
|------------|------------|----------------------------------------------|
| identifier | required   | A unique identifier of this type             |
| name       | required   | The human-readable name of this type         |
| color      | (optional) | The color of this type                       |
| controlled | (optional) | If present, a value can be entered via the UI |


The UI additionally supports special input controls via the `controlled` parameter.
By default, no special input controls are added.

| Controlled=             | Description                                                 |
|-------------------------|-------------------------------------------------------------|
| text                    | A generic text input                                        |
| number                  | A numeric text input                                        |
| password                | A generic text input with masked inputs                     |
| boolean                 | A boolean input                                             |
| RemoteFile\|F1\|...\|Fn | A selection of a file present in the codebase               |
| Select\|O1\|...\|On     | A drop-down with the Options O1...On                        |
| Table\|C1\|...\|Cn      | A table with the columns C1...Cn and any number of rows     |
| List\|C1                | A list with the one C1 and any number of rows               |
| Tuple\|C1\|...\|Cn      | A table with only one row                                   |
| _(internal)_            | See type `BasicallyABufferedFile` for uploading files       |
| _(internal)_            | See Dependant Inputs/SmartInputs; another layer of controls |

To add support for additional input types,
they have to be added to the UI (feel free to ask Alex for this).

#### Defining more complex types

For types, that are not built-ins, you can define your own classes.
We use pydantic (dataclasses or BaseModel) to define types.

````python
from pydantic.dataclasses import dataclass
from py_api_wbgeo.nodesapi import wbgeo_type

@wbgeo_type(name='Input data for a geological model', color='orange',
            identifier='wbgeo::my_complex_data_type')
@dataclass
class MyComplexDataType:
   name: str
   numbers: MyListOfNumbers

````

You must not use pydantics 'arbitrary_types' config option.
Take a look at the pydantic_bridge adapters instead.

