# Codebase Installation

This document outlines an installation guide for using the workbench directly
in Python scripts or notebooks.

See the [visual editor installation](visual.md) instead if you want to run
WBGeo through its browser-based interface in Docker.

## Requirements

WBGeo requires Python 3.12 or later. We recommend installing into a virtual
environment, e.g.:

```
python -m venv .venv
.venv\Scripts\activate   # on Windows
source .venv/bin/activate   # on Linux/macOS
```

On Windows, the [sfepy](https://sfepy.org/doc/installation.html) package (used
for process simulation) needs a C compiler to build during install. If you
don't plan to run simulations and don't want to install a compiler, clone the
repository, remove the `sfepy` line from `requirements.txt`, and install from
that local checkout instead of the git URL above:

```
git clone https://github.com/wbgeo/codebase.git
cd codebase
# remove the sfepy line from requirements.txt, then:
pip install .
```

## Install via pip

The core packages can be installed directly from source via:

> `pip install git+https://github.com/wbgeo/codebase.git`

## Test Installation

You can test your installation by running one of the examples. Since the
examples are not part of the pip-installed package, checkout just the
`examples/` folder via a sparse checkout:

```
git clone --filter=blob:none --no-checkout https://github.com/wbgeo/codebase.git
cd codebase
git sparse-checkout set examples
git checkout main
```

Then run one of the models, e.g.:

> `python examples/synthetic_examples/model1/WBGeo1.0_model1.py`
