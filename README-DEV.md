# Confusion Matrix in Python
This file purpose is for contributors support or someone that want to improve de library.
For information about using the library pretty-print-confusion see [README.md](README.md)  

## Environment Dependencies
- python ">=3.10,<3.14"
- poetry
- git
- 

## Setup local dev environment
- Clone and open dir using `git clone git@github.com:wcipriano/pretty-print-confusion-matrix.git && cd pretty-print-confusion-matrix`
- Create virtual environment (venv) like this example `python3.11 -m venv .venv`
- Activate venv `source ./.venv/bin/activate
- Lock dependencies with `poetry lock` 
- Install dependencies with `poetry install`
- Enjoy it \o/ \o/ \o/

## Test
- Test and coverage with `poetry run pytest --cov=./ --cov-branch --cov-report=xml`

## Code formatter
- Check quality of code (format) with `poetry run black . --check`


## References:

### Project URLs
- [Github](https://pypi.org/project/pretty-confusion-matrix/)
- [PYPI](https://pypi.org/project/pretty-confusion-matrix/)
- 


### Python packaging and dependency management

1. [Poetry docs](https://python-poetry.org/docs/)
2. [Poetry dependency specification](https://python-poetry.org/docs/dependency-specification/)
3. Statistics and dependency status by [Libraries.io](https://libraries.io/pypi/pretty-confusion-matrix)
4. 
