from pathlib import Path

import mrcfile
import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import (
    reconstruct_subvolume,
    reconstruct_subvolume_from_tilt_series,
    reconstruct_tomogram_from_tilt_series,
)

# Choose device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pixel spacing in Angstroms (required)
PIXEL_SPACING = 1.548

# --- 1. Load a tilt series -------------------------------------------------
# From AreTomo output (.aln file + tilt stack):
ALN_PATH = Path("/path/to/your/alignment.aln")
TILT_STACK_PATH = Path("/path/to/your/tilt_stack.mrc")
tilt_series = TiltSeries.from_aretomo_output(
    aln_path=ALN_PATH,
    pixel_spacing=PIXEL_SPACING,
    image_path=TILT_STACK_PATH,
    device=DEVICE,
)

# Or from an ETOMO directory:
# tilt_series = TiltSeries.from_etomo_directory(
#     etomo_dir=Path("/path/to/etomo/dir"),
#     pixel_spacing=PIXEL_SPACING,
#     device=DEVICE,
# )

# --- 2. Inspect geometry ---------------------------------------------------
print("images", tuple(tilt_series.images.shape))
print("projection_matrices", tuple(tilt_series.projection_matrices.shape))

# Move between devices if needed
tilt_series.to(DEVICE)

# --- 3. Project 3D points into the tilt images -----------------------------
# Points are zyx, in Angstroms, relative to the tomogram center.
points_zyx = torch.tensor([[0.0, 0.0, 0.0], [100.0, -50.0, 200.0]], device=DEVICE)
projected_yx = tilt_series.project_points(points_zyx)  # (n_points, n_tilts, 2)

# --- 4. Extract a subtilt-series around each point -------------------------
particle_tilt_series = tilt_series.extract_particle_tilt_series(
    points_zyx, sidelength=64, return_rfft=False
)  # (n_points, n_tilts, 64, 64)

# --- 5. Reconstruct subvolumes at the points -------------------------------
# torch-reconstruct-tomogram has no dependency on TiltSeries: it takes plain
# images / projection_matrices / pixel_spacing tensors...
subvolumes = reconstruct_subvolume(
    tilt_series.images, tilt_series.projection_matrices, tilt_series.pixel_spacing,
    points_zyx, sidelength=64,
)  # (n_points, 64, 64, 64)

# ...or, equivalently, the *_from_tilt_series() convenience wrappers can be used
# directly on a TiltSeries (or anything with .images/.projection_matrices/pixel_spacing)
subvolumes = reconstruct_subvolume_from_tilt_series(
    tilt_series, points_zyx, sidelength=64
)  # (n_points, 64, 64, 64)

# --- 6. Reconstruct the full tomogram --------------------------------------
volume_shape = (256, 512, 512)
sidelength = 128
tomogram = reconstruct_tomogram_from_tilt_series(
    tilt_series, volume_shape, sidelength, batch_size=None
)

# --- 7. Save the result ----------------------------------------------------
output_path = ALN_PATH.parent / "torch_tomogram_reconstruction.mrc"
mrcfile.write(
    output_path,
    tomogram.cpu().numpy(),
    overwrite=True,
    voxel_size=PIXEL_SPACING,
)
