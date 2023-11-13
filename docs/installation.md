# fitburst - Installation

## Building from `PyPI`

`fitburst` will soon be retrievable from the Python Package Index ([PyPI](https://pypi.org)) via [pip](https://pypi.org/project/pip/).

## Building from Source

The `fitburst` codebase uses either `pip` or [poetry](https://python-poetry.org) for building the distribution and grabbing dependencies. The easiest route to building `fitburst` is to use `pip`:

    > git clone https://github.com/CHIMEFRB/fitburst.git
    > cd fitburst/
    > pip install .

## Dependencies
For out-of-the-box use, `fitburst` currently depends on the external Python packages listed below. We encourage interested developers to contribute software and/or replace existing functionality with new dependencies; however, we request that any additional dependency be open-source, meaningfully used, and accessible via `pip`.

* Python 3.8 or greater
* [matplotlib](https://matplotlib.org/users/installing.html)
* [mkdocs](https://www.mkdocs.org/#installation)
* [numpy](https://numpy.org/install/)
* [pandas](https://pandas.pydata.org/docs/index.html)
* [pytz](https://pythonhosted.org/pytz/)
* [pyyaml](https://pyyaml.org/wiki/PyYAMLDocumentation)
* [scipy](https://www.scipy.org/install.html)

### Developer Dependencies
There are additional dependencies for folks who wish to contribute and build code, tests, and/or documentation. These dependenices are listed in the `pyproject.toml` file under the `tools.poetry.group.*.dependencies` attributes. However, these dependencies currently can only be installed using `poetry` due to way in which `pip` understands the `pyproject.toml` file.
