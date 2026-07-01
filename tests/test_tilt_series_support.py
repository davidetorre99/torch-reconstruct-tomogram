import torch
from torch_tilt_series import TiltSeries

from torch_reconstruct_tomogram import (
    reconstruct_subvolume,
    reconstruct_subvolume_from_tilt_series,
    reconstruct_tomogram,
    reconstruct_tomogram_from_tilt_series,
)


def make_tilt_series(size=32):
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    tilt_axis_angle = torch.tensor(0.0)
    sample_translations = torch.zeros((3, 2))
    images = torch.zeros((3, size, size))
    c = size // 2
    images[:, c - 2:c + 2, c - 2:c + 2] = 1.0
    return TiltSeries(
        tilt_angles=tilt_angles,
        tilt_axis_angle=tilt_axis_angle,
        sample_translations=sample_translations,
        images=images,
        pixel_spacing=1.0,
    )


def test_reconstruct_subvolume_from_tilt_series_matches_explicit_tensors():
    ts = make_tilt_series()
    point_zyx = torch.tensor([0.0, 0.0, 0.0])

    from_wrapper = reconstruct_subvolume_from_tilt_series(ts, point_zyx, sidelength=8)
    explicit = reconstruct_subvolume(
        ts.images, ts.projection_matrices, ts.pixel_spacing, point_zyx, sidelength=8
    )
    assert torch.equal(from_wrapper, explicit)


def test_reconstruct_tomogram_from_tilt_series_matches_explicit_tensors():
    ts = make_tilt_series()

    from_wrapper = reconstruct_tomogram_from_tilt_series(ts, (16, 16, 16), sidelength=8)
    explicit = reconstruct_tomogram(
        ts.images, ts.projection_matrices, ts.pixel_spacing, (16, 16, 16), sidelength=8
    )
    assert torch.equal(from_wrapper, explicit)
