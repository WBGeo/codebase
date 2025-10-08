# Codebase

First repository in WBgeo to organize the codebase.

## Structure

- `core/`: Components
- `examples/`: Example scripts
- `examples/data/`: Input data for example models
- `concepts/`: Concepts and ideas for future developments

## Explicit Component Definition

The

`````python
from py_api_wbgeo.nodesapi import *
`````

An example component can be defined like the following:

```python
@wbgeo_component(identifier='py::create_random_number',
                 title='Create Random Number',
                 description='Create a random number', color='#03b1fc')
def create_random_number() -> int:
   return random.randint(0, 100)
```

It is decorated via the `@wbgeo_component` decorator,
which MUST contain a unique identifier, a human-readable title and description,
as well as an optional color.
The function's signature MUST be explicit, i.e., contain a returned type &
parameters.
