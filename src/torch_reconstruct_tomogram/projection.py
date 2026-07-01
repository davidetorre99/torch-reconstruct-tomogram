"""Project 3D points into tilt images and crop patches for reconstruction."""

from collections.abc import Callable

import einops
import torch
from torch_affine_utils import homogenise_coordinates
from torch_grid_utils import dft_center
from torch_subpixel_crop import subpixel_crop_2d

LocalShiftFn = Callable[[torch.Tensor], torch.Tensor]


def project_points(
    points_zyx: torch.Tensor,
    projection_matrices: torch.Tensor,
    pixel_spacing: float,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Project 3D points to 2D image coordinates.

    - points are 3D zyx coordinates in Angstroms, relative to the tomogram center
    - projection_matrices are (n_tilts, 4, 4) homogeneous zyx -> yx matrices
    - projected 2D points are in pixels, relative to the center of each image
    - local_shifts, if provided, is called with the projected points
      (n_points, n_tilts, 2) and must return a correction of the same shape,
      which is added to the projected points before returning
    """
    points_zyx = torch.as_tensor(
        points_zyx, device=projection_matrices.device
    ).float()

    # Convert from Angstroms to pixels for projection
    points_zyx_px = points_zyx / pixel_spacing

    # Apply projection matrices
    M_yx = projection_matrices[..., [1, 2], :]  # (ntilts, 2, 4)
    points_zyxw = homogenise_coordinates(points_zyx_px)
    projected_yx = M_yx @ einops.rearrange(
        points_zyxw, "nparticles zyxw -> nparticles 1 zyxw 1"
    )
    projected_yx = einops.rearrange(
        projected_yx, "nparticles ntilts yx 1 -> nparticles ntilts yx"
    )
    if local_shifts is not None:
        projected_yx = projected_yx + local_shifts(projected_yx)
    return projected_yx  # (points, tilts, yx)


def extract_particle_tilt_series(
    images: torch.Tensor,
    points_zyx: torch.Tensor,
    projection_matrices: torch.Tensor,
    pixel_spacing: float,
    sidelength: int,
    return_rfft: bool = True,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Extract a subtilt-series at 3D location(s) in the sample."""
    projected_yx = project_points(
        points_zyx, projection_matrices, pixel_spacing, local_shifts=local_shifts
    )
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
