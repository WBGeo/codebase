


## Adding new components
See the [developers guide](DevelopersGuide.md) for a description 
on how components can be defined and what rules are to be followed.

TOOD: Description on the idea/design-vision of the workbench (accessibility)
--Best-practice on that here--
(direct input vs load-like extra component)

(Feel free to talk to Jan von Harten/the entire team to check if the visions are compatible :) )

For each component (added):
   * Follow the naming guide for the identifier
   * Use pydocs explaining your component and its inputs
   * Add a unit test of the execution 
      * use a small input to keep it fast
      * See the [test_rbf_interpolation.py](test/interpolator_components/test_rbf_interpolation.py) unit test for an example of this
   * Consider invalid inputs/limitations and define pre-checks for them
     * Add unit tests for each pre-check
       * Each pre-check MUST be tested
   * Add the new component to an example (python) workflow
     * And ensure the workflow runs (on your machine)


## Adding new types
See the [developers guide](DevelopersGuide.md) for a description
on how types can be defined and what rules are to be followed.

For each type (added):
   * Follow the naming guide
   * Use pydocs explaining your type
   * Add a unit test ensuring it fulfills the following requirements (TODO: example):
      * a valid pydantic model
      * serializable and deserializable 
   * Use it in an example and run the example on your machine

Note: pydantic does not support some types, such as numpy arrays, without annotated
 information on how to (de)serialize them.
For numpy types, please use [pydantic_numpy](https://pypi.org/project/pydantic_numpy/) 
 and see the MeshResults [here](core/object_components.py) for additional 
 examples and here (TODO) for the pydantic docs on that topic.


