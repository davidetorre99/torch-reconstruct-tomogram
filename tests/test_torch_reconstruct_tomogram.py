import pytest
import torch
from torch_tilt_series import TiltSeries

import torch_reconstruct_tomogram
from torch_reconstruct_tomogram import reconstruct_subvolume, reconstruct_tomogram

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_tilt_series(device="cpu", size=32):
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
        device=device,
    )


def test_imports_with_version():
    assert isinstance(torch_reconstruct_tomogram.__version__, str)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume(device):
    ts = make_tilt_series(device)
    point_zyx = torch.tensor([0.0, 0.0, 0.0], device=device)
    subvolume = reconstruct_subvolume(ts, point_zyx, sidelength=8)
    assert subvolume.shape == (8, 8, 8)
    assert subvolume.dtype == torch.float32
    assert device in str(subvolume.device)
    assert torch.isfinite(subvolume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_rank_polymorphic(device):
    ts = make_tilt_series(device)

    # single point (3,) -> (d, h, w)
    point = torch.tensor([0.0, 0.0, 0.0])
    assert reconstruct_subvolume(ts, point, sidelength=8).shape == (8, 8, 8)

    # batch (N, 3) -> (N, d, h, w)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    assert reconstruct_subvolume(ts, points, sidelength=8).shape == (2, 8, 8, 8)

    # 2D grid (a, b, 3) -> (a, b, d, h, w)
    grid_2d = torch.zeros(2, 3, 3)
    assert reconstruct_subvolume(ts, grid_2d, sidelength=8).shape == (2, 3, 8, 8, 8)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_output_pixel_spacing(device):
    ts = make_tilt_series(device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    subvolume_default = reconstruct_subvolume(ts, point, sidelength=8)
    subvolume_explicit = reconstruct_subvolume(
        ts, point, sidelength=8, output_pixel_spacing=ts.pixel_spacing
    )
    assert torch.allclose(subvolume_default, subvolume_explicit)

    subvolume_coarse = reconstruct_subvolume(
        ts, point, sidelength=8, output_pixel_spacing=2.0
    )
    assert subvolume_coarse.shape == (8, 8, 8)
    assert torch.isfinite(subvolume_coarse).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_output_pixel_spacing(device):
    ts = make_tilt_series(device)
    volume = reconstruct_tomogram(
        ts, (16, 16, 16), sidelength=8, output_pixel_spacing=2.0
    )
    assert volume.shape == (16, 16, 16)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram(device):
    ts = make_tilt_series(device)
    volume = reconstruct_tomogram(ts, (16, 16, 16), sidelength=8)
    assert volume.shape == (16, 16, 16)
    assert volume.dtype == torch.float32
    assert device in str(volume.device)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_non_cubic(device):
    ts = make_tilt_series(device)
    # shape not divisible by sidelength is still cropped to the requested shape
    volume = reconstruct_tomogram(ts, (8, 24, 20), sidelength=8)
    assert volume.shape == (8, 24, 20)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_batch_size(device):
    ts = make_tilt_series(device)
    recon_no_batch = reconstruct_tomogram(ts, (16, 16, 16), sidelength=8)
    recon_with_batch = reconstruct_tomogram(
        ts, (16, 16, 16), sidelength=8, batch_size=2
    )
    assert recon_no_batch.shape == (16, 16, 16)
    assert recon_with_batch.shape == (16, 16, 16)
    diff = torch.abs(recon_no_batch - recon_with_batch.to(recon_no_batch.device)).max()
    assert diff == 0.0
