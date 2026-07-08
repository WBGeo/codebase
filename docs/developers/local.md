# Local execution
## Running the examples locally

You can either use Intellij with its python console to run the examples,
or modify your `PYTHONPATH` environment variable:

````bash
# linux
cd codebase
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/synthetic_examples/model1/WBGeo1.0_model1.py
````

````powershell
# powershell
cd codebase
$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH
python examples/synthetic_examples/model1/WBGeo1.0_model1.py
````

## Running the visual frontend locally

See the [visual editor installation guide](../installation/visual.md) for how to use
a local codebase.
