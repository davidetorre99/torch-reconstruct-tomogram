"""(sub-)tomogram reconstruction in pytorch."""

import einops
import torch
import torch.nn.functional as F
from torch_fourier_rescale import fourier_rescale_3d
from torch_fourier_slice import insert_central_slices_rfft_3d_multichannel
from torch_grid_utils import fftfreq_grid

from torch_reconstruct_tomogram.projection import (
    LocalShiftFn,
    extract_particle_tilt_series,
)

_PAD_FACTOR = 2.0


def _reconstruct_subvolume_at_input_spacing(
    images: torch.Tensor,
    projection_matrices: torch.Tensor,
    pixel_spacing: float,
    points_zyx: torch.Tensor,
    sidelength: int,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    device = images.device
    points_zyx = torch.as_tensor(points_zyx, device=device).float()

    points_zyx, ps = einops.pack([points_zyx], "* zyx")

    rotation_matrices = projection_matrices[:, :3, :3]
    rotation_matrices = torch.linalg.pinv(rotation_matrices)
    sidelength_padded = int(_PAD_FACTOR * sidelength)

    particle_tilt_series_rfft = extract_particle_tilt_series(
        images,
        points_zyx,
        projection_matrices,
        pixel_spacing,
        sidelength=sidelength_padded,
        return_rfft=True,
        local_shifts=local_shifts,
    )

    particle_tilt_series_rfft = torch.fft.fftshift(
        particle_tilt_series_rfft, dim=(-2,)
    )

    particle_tilt_series_rfft = einops.rearrange(
        particle_tilt_series_rfft,
        "n_positions n_tilts h w_rfft -> n_tilts n_positions h w_rfft"
    )

    patches_rfft, weights = insert_central_slices_rfft_3d_multichannel(
        image_rfft=particle_tilt_series_rfft,
        volume_shape=(sidelength_padded, sidelength_padded, sidelength_padded),
        rotation_matrices=rotation_matrices,
        zyx_matrices=True,
        fftfreq_max=0.5,
    )

    valid_weights = weights > 1e-3
    patches_rfft[:, valid_weights] /= weights[valid_weights]

    patches_rfft = torch.fft.ifftshift(patches_rfft, dim=(-3, -2))

    patches = torch.fft.irfftn(
        patches_rfft,
        s=(sidelength_padded,) * 3,
        dim=(-3, -2, -1)
    )

    patches = torch.fft.ifftshift(patches, dim=(-3, -2, -1))

    grid = fftfreq_grid(
        image_shape=(sidelength_padded, sidelength_padded, sidelength_padded),
        rfft=False, fftshift=True, norm=True, device=device
    )
    patches = patches / torch.sinc(grid) ** 2

    p = (sidelength_padded - sidelength) // 2
    patches = F.pad(patches, [-p] * 6)

    [patches] = einops.unpack(patches, ps, "* d h w")

    return patches


def reconstruct_subvolume(
    images: torch.Tensor,
    projection_matrices: torch.Tensor,
    pixel_spacing: float,
    points_zyx: torch.Tensor,
    sidelength: int,
    output_pixel_spacing: float | None = None,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Reconstruct 3D patch(es) at location(s) in the sample.

    Rank-polymorphic: input (..., 3) -> output (..., d, h, w)

    - images is the tilt-series stack, shape (n_tilts, h, w)
    - projection_matrices are (n_tilts, 4, 4) homogeneous zyx -> yx matrices
      mapping 3D sample positions (pixels, tomogram-center-relative) to 2D
      image positions (pixels, image-center-relative)
    - pixel_spacing is the voxel size of images/projection_matrices, in Angstroms
    - points_zyx are zyx coordinates in Angstroms, relative to the tomogram center
    - sidelength is the output subvolume size in voxels
    - output_pixel_spacing is the voxel size of the output in Angstroms (defaults to
      pixel_spacing)
    - local_shifts, if provided, is called with projected 2D positions
      (n_points, n_tilts, 2) and must return a pixel-space correction of the same
      shape (e.g. from patch-based local alignment)

    See also `reconstruct_subvolume_from_tilt_series` for a convenience wrapper
    that takes a `torch_tilt_series.TiltSeries` directly.
    """
    input_pixel_spacing = pixel_spacing
    if output_pixel_spacing is None:
        output_pixel_spacing = input_pixel_spacing

    if output_pixel_spacing == input_pixel_spacing:
        return _reconstruct_subvolume_at_input_spacing(
            images,
            projection_matrices,
            pixel_spacing,
            points_zyx,
            sidelength,
            local_shifts=local_shifts,
        )

    sidelength_input = max(
        1, round(sidelength * output_pixel_spacing / input_pixel_spacing)
    )
    patches = _reconstruct_subvolume_at_input_spacing(
        images,
        projection_matrices,
        pixel_spacing,
        points_zyx,
        sidelength_input,
        local_shifts=local_shifts,
    )
    target_shape = (sidelength, sidelength, sidelength)
    patches, _ = fourier_rescale_3d(
        patches,
        source_spacing=input_pixel_spacing,
        target_shape=target_shape,
    )
    return patches


def reconstruct_tomogram(
    images: torch.Tensor,
    projection_matrices: torch.Tensor,
    pixel_spacing: float,
    volume_shape: tuple[int, int, int],
    sidelength: int,
    batch_size: int | None = None,
    output_pixel_spacing: float | None = None,
    local_shifts: LocalShiftFn | None = None,
) -> torch.Tensor:
    """Reconstruct the full tomogram by tiling reconstructed patches in 3D.

    See also `reconstruct_tomogram_from_tilt_series` for a convenience wrapper
    that takes a `torch_tilt_series.TiltSeries` directly.
    """
    if output_pixel_spacing is None:
        output_pixel_spacing = pixel_spacing

    d, h, w = volume_shape
    r = sidelength // 2

    device = images.device
    z = torch.arange(start=r, end=d + r, step=sidelength, device=device) - d // 2
    y = torch.arange(start=r, end=h + r, step=sidelength, device=device) - h // 2
    x = torch.arange(start=r, end=w + r, step=sidelength, device=device) - w // 2

    centers_zyx = torch.stack(
        torch.meshgrid(z, y, x, indexing='ij'), dim=-1
    )

    centers_zyx_ang = centers_zyx * output_pixel_spacing

    if batch_size is None:
        patches = reconstruct_subvolume(
            images,
            projection_matrices,
            pixel_spacing,
            points_zyx=centers_zyx_ang,
            sidelength=sidelength,
            output_pixel_spacing=output_pixel_spacing,
            local_shifts=local_shifts,
        )
        tomogram = einops.rearrange(
            patches,
            'gd gh gw d h w -> (gd d) (gh h) (gw w)'
        )
    else:
        gd, gh, gw = centers_zyx.shape[:3]
        tomogram_shape = (gd * sidelength, gh * sidelength, gw * sidelength)
        tomogram = torch.zeros(tomogram_shape, device='cpu', dtype=torch.float32)

        centers_flat, _ = einops.pack([centers_zyx_ang], "* zyx")
        total_patches = centers_flat.shape[0]

        patch_indices = torch.arange(total_patches)
        iz_all = patch_indices // (gh * gw)
        iy_all = (patch_indices % (gh * gw)) // gw
        ix_all = patch_indices % gw

        batch_idx = 0
        for chunk in centers_flat.split(batch_size):
            patches_batch = reconstruct_subvolume(
                images,
                projection_matrices,
                pixel_spacing,
                chunk,
                sidelength,
                output_pixel_spacing=output_pixel_spacing,
                local_shifts=local_shifts,
            ).cpu()

            for j in range(len(patches_batch)):
                idx = batch_idx + j
                iz, iy, ix = iz_all[idx], iy_all[idx], ix_all[idx]

                tomogram[
                    iz*sidelength:(iz+1)*sidelength,
                    iy*sidelength:(iy+1)*sidelength,
                    ix*sidelength:(ix+1)*sidelength
                ] = patches_batch[j]

            batch_idx += len(patches_batch)

            del patches_batch
            if device.type != 'cpu':
                torch.cuda.empty_cache()

    tomogram = tomogram[:d, :h, :w]

    return tomogram
