# torch-reconstruct-tomogram

[![License](https://img.shields.io/pypi/l/torch-reconstruct-tomogram.svg?color=green)](https://github.com/teamtomo/torch-reconstruct-tomogram/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/torch-reconstruct-tomogram.svg?color=green)](https://pypi.org/project/torch-reconstruct-tomogram)
[![Python Version](https://img.shields.io/pypi/pyversions/torch-reconstruct-tomogram.svg?color=green)](https://python.org)
[![CI](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml/badge.svg)](https://github.com/teamtomo/torch-reconstruct-tomogram/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram/branch/main/graph/badge.svg)](https://codecov.io/gh/teamtomo/torch-reconstruct-tomogram)

(sub-)Tomogram reconstruction and subtilt extraction for cryo-ET.

## Overview

This package provides (sub-)tomogram reconstruction from a tilt series. It supports

* `reconstruct_subvolume()` - rank-polymorphic reconstruction of 3D patch(es) at location(s) in the sample
* `reconstruct_tomogram()` - full volume reconstruction by tiling reconstructed patches in 3D

Both take plain tensors: `images` (the tilt-series stack), `projection_matrices` (per-tilt homogeneous zyx -> yx matrices), and `pixel_spacing`. Reconstruction is performed in Fourier space using central slice insertion. Positions are in `zyx` coordinates, in Angstroms, relative to the tomogram center.

If you already have a [`torch-tilt-series`](https://github.com/teamtomo/torch-tilt-series) `TiltSeries` (or any object exposing `.images`, `.projection_matrices` and `.pixel_spacing`), the `reconstruct_subvolume_from_tilt_series()` / `reconstruct_tomogram_from_tilt_series()` wrappers skip unpacking those attributes. 

## Installation

```bash
pip install torch-reconstruct-tomogram
```

To load a tilt series from AreTomo or ETOMO output and use the `*_from_tilt_series()` wrappers, also install  [`torch-tilt-series`](https://github.com/teamtomo/torch-tilt-series):

```bash
pip install torch-tilt-series[io]
```

## Examples

See the [`examples/`](examples/) folder for scripts showing how to load a tilt series, reconstruct subvolumes and tomograms, and save the result.

## License

This project is licensed under the BSD 3-Clause License - see the LICENSE file for details.
