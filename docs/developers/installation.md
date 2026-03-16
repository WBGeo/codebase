# Developer Installation

The (internal python) DSL is available as a decorator-based API via pip at
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

