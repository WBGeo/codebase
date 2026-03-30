# Running the examples locally
You can either use Intellij with its python console to run the examples,
or modify your `PYTHONPATH` environment variable:

````bash
# linux
cd codebase
export PYTHONPATH="$PWD:$PYTHONPATH"
python examples/WBGeo123.py
````

````powershell
# powershell
cd codebase
$env:PYTHONPATH = "$PWD;" + $env:PYTHONPATH
python examples/WBGeo123.py
````

# Running the visual frontend locally

TODO

* Note: Currently a runner does not update existing components or type definitions. 
You can either restart the backend or click the button on the `/gui/Reset/Reset` page.
