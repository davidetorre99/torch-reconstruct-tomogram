# torch-reconstruct-tomogram

[![License](https://img.shields.io/pypi/l/torch-reconstruct-tomogram.svg?color=green)](https://github.com/teamtomo/torch-reconstruct-tomogram/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-reconstruct-tomogram.svg?color=green)](https://pypi.org/project/torch-reconstruct-tomogram)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-reconstruct-tomogram.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram)

(sub-)tomogram reconstruction and subtilt extraction for cryoET.

## Development

The easiest way to get started is to use the [github cli](https://cli.github.com)
and [uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
gh repo fork teamtomo/torch-reconstruct-tomogram --clone
# or just
# gh repo clone teamtomo/torch-reconstruct-tomogram
cd torch-reconstruct-tomogram
uv sync
```

Run tests:

```sh
uv run pytest
```

Lint files:

```sh
uv run pre-commit run --all-files
```
