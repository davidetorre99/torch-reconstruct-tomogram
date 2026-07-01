"""wrapper for reconstructing from a tilt-series-like object.

These do not import or depend on `torch-tilt-series`. Any object exposing
`.images`, `.projection_matrices` and `.pixel_spacing` with the same semantics
as `torch_tilt_series.TiltSeries` works here (`TiltSeries` itself included).
"""

from typing import Protocol

import torch

from torch_reconstruct_tomogram.projection import LocalShiftFn
from torch_reconstruct_tomogram.reconstruct import (
    reconstruct_subvolume,
    reconstruct_tomogram,
)


class TiltSeriesLike(Protocol):
    """Structural type for objects reconstruction can be driven from."""

    images: torch.Tensor
    projection_matrices: torch.Tensor
    pixel_spacing: float


def reconstruct_subvolume_from_tilt_series(
    tilt_series: TiltSeriesLike,
    points_zyx: torch.Tensor,
    sidelength: int,
    output_pixel_spacing: float | None = None,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Reconstruct 3D patch(es) from a tilt-series-like object.

    See `reconstruct_subvolume` for argument details.
    """
    return reconstruct_subvolume(
        tilt_series.images,
        tilt_series.projection_matrices,
        tilt_series.pixel_spacing,
        points_zyx,
        sidelength,
        output_pixel_spacing=output_pixel_spacing,
        local_shifts=local_shifts,
    )


def reconstruct_tomogram_from_tilt_series(
    tilt_series: TiltSeriesLike,
    volume_shape: tuple[int, int, int],
    sidelength: int,
    batch_size: int | None = None,
    output_pixel_spacing: float | None = None,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Reconstruct the full tomogram from a tilt-series-like object.

    See `reconstruct_tomogram` for argument details.
    """
    return reconstruct_tomogram(
        tilt_series.images,
        tilt_series.projection_matrices,
        tilt_series.pixel_spacing,
        volume_shape,
        sidelength,
        batch_size=batch_size,
        output_pixel_spacing=output_pixel_spacing,
        local_shifts=local_shifts,
    )
