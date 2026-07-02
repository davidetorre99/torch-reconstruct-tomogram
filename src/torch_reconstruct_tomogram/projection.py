"""Project 3D points into tilt images and crop patches for reconstruction."""

from collections.abc import Callable

import torch
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram.io import (
    load_tilt_series_images,
    normalize_on_central_crop,
)

LocalShiftFn = Callable[[torch.Tensor], torch.Tensor]


def _require_pixel_spacing(tilt_series: TiltSeries) -> float:
    if tilt_series.pixel_spacing is None:
        raise ValueError(
            "tilt_series.pixel_spacing is not set -> construct the TiltSeries "
            "via a torch_tilt_series loader (e.g. from_aretomo_output, "
            "from_etomo_directory), or set it yourself."
        )
    return tilt_series.pixel_spacing


def project_points(
    tilt_series: TiltSeries,
    points_zyx: torch.Tensor,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Project 3D points to 2D image pixel coordinates.

    - points are 3D zyx coordinates in Angstroms, relative to the tomogram center
    - tilt_series supplies the projection geometry (`tilt_series.project_points`
      works in Angstroms) and `tilt_series.pixel_spacing`, used to convert the
      projected Angstrom positions to pixels
    - projected 2D points are in pixels, relative to the center of each image
    - local_shifts, if provided, is called with the projected points
      (n_points, n_tilts, 2) and must return a correction of the same shape,
      which is added to the projected points before returning
    """
    pixel_spacing = _require_pixel_spacing(tilt_series)
    projected_yx_ang = tilt_series.project_points(points_zyx)
    projected_yx = projected_yx_ang / pixel_spacing
    if local_shifts is not None:
        projected_yx = projected_yx + local_shifts(projected_yx)
    return projected_yx  # (points, tilts, yx)


def _extract_particle_tilt_series(
    tilt_series: TiltSeries,
    images: torch.Tensor,
    points_zyx: torch.Tensor,
    sidelength: int,
    return_rfft: bool = True,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Extract a subtilt-series given already-loaded images."""
    projected_yx = project_points(tilt_series, points_zyx, local_shifts=local_shifts)
    projected_yx = projected_yx + dft_center(
        images.shape[-2:], rfft=False, fftshift=True, device=images.device
    )
    return subpixel_crop_2d(
        image=images,
        positions=projected_yx,
        sidelength=sidelength,
        return_rfft=return_rfft,
        decenter=return_rfft,
    )


def extract_particle_tilt_series(
    tilt_series: TiltSeries,
    points_zyx: torch.Tensor,
    sidelength: int,
    return_rfft: bool = True,
    normalize: bool = True,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Extract a subtilt-series at 3D location(s) in the sample.

    Loads (and, by default, normalizes) the raw tilt images matching
    `tilt_series` via `tilt_series.image_path`/`image_indices`.
    """
    images = load_tilt_series_images(tilt_series)
    if normalize:
        images = normalize_on_central_crop(images)
    return _extract_particle_tilt_series(
        tilt_series,
        images,
        points_zyx,
        sidelength,
        return_rfft=return_rfft,
        local_shifts=local_shifts,
    )
