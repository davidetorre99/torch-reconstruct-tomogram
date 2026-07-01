import pytest
import torch
from torch_affine_utils.transforms_3d import Ry

import torch_reconstruct_tomogram
from torch_reconstruct_tomogram import reconstruct_subvolume, reconstruct_tomogram

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def make_geometry(device="cpu", size=32):
    tilt_angles = torch.tensor([-30.0, 0.0, 30.0])
    images = torch.zeros((3, size, size), device=device)
    c = size // 2
    images[:, c - 2:c + 2, c - 2:c + 2] = 1.0
    projection_matrices = Ry(tilt_angles, zyx=True, device=device)
    return images, projection_matrices


def test_imports_with_version():
    assert isinstance(torch_reconstruct_tomogram.__version__, str)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume(device):
    images, projection_matrices = make_geometry(device)
    point_zyx = torch.tensor([0.0, 0.0, 0.0], device=device)
    subvolume = reconstruct_subvolume(
        images, projection_matrices, 1.0, point_zyx, sidelength=8
    )
    assert subvolume.shape == (8, 8, 8)
    assert subvolume.dtype == torch.float32
    assert device in str(subvolume.device)
    assert torch.isfinite(subvolume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_rank_polymorphic(device):
    images, projection_matrices = make_geometry(device)

    # single point (3,) -> (d, h, w)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)
    assert reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8
    ).shape == (8, 8, 8)

    # batch (N, 3) -> (N, d, h, w)
    points = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], device=device)
    assert reconstruct_subvolume(
        images, projection_matrices, 1.0, points, sidelength=8
    ).shape == (2, 8, 8, 8)

    # 2D grid (a, b, 3) -> (a, b, d, h, w)
    grid_2d = torch.zeros(2, 3, 3, device=device)
    assert reconstruct_subvolume(
        images, projection_matrices, 1.0, grid_2d, sidelength=8
    ).shape == (2, 3, 8, 8, 8)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_output_pixel_spacing(device):
    images, projection_matrices = make_geometry(device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    subvolume_default = reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8
    )
    subvolume_explicit = reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8,
        output_pixel_spacing=1.0,
    )
    assert torch.allclose(subvolume_default, subvolume_explicit)

    subvolume_coarse = reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8,
        output_pixel_spacing=2.0,
    )
    assert subvolume_coarse.shape == (8, 8, 8)
    assert torch.isfinite(subvolume_coarse).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_subvolume_local_shifts(device):
    images, projection_matrices = make_geometry(device)
    point = torch.tensor([0.0, 0.0, 0.0], device=device)

    def zero_local_shifts(projected_yx):
        return torch.zeros_like(projected_yx)

    subvolume_default = reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8
    )
    subvolume_with_hook = reconstruct_subvolume(
        images, projection_matrices, 1.0, point, sidelength=8,
        local_shifts=zero_local_shifts,
    )
    assert torch.allclose(subvolume_default, subvolume_with_hook)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_output_pixel_spacing(device):
    images, projection_matrices = make_geometry(device)
    volume = reconstruct_tomogram(
        images, projection_matrices, 1.0, (16, 16, 16), sidelength=8,
        output_pixel_spacing=2.0,
    )
    assert volume.shape == (16, 16, 16)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram(device):
    images, projection_matrices = make_geometry(device)
    volume = reconstruct_tomogram(
        images, projection_matrices, 1.0, (16, 16, 16), sidelength=8
    )
    assert volume.shape == (16, 16, 16)
    assert volume.dtype == torch.float32
    assert device in str(volume.device)
    assert torch.isfinite(volume).all()


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_non_cubic(device):
    images, projection_matrices = make_geometry(device)
    # shape not divisible by sidelength is still cropped to the requested shape
    volume = reconstruct_tomogram(
        images, projection_matrices, 1.0, (8, 24, 20), sidelength=8
    )
    assert volume.shape == (8, 24, 20)


@pytest.mark.parametrize("device", DEVICES)
def test_reconstruct_tomogram_batch_size(device):
    images, projection_matrices = make_geometry(device)
    recon_no_batch = reconstruct_tomogram(
        images, projection_matrices, 1.0, (16, 16, 16), sidelength=8
    )
    recon_with_batch = reconstruct_tomogram(
        images, projection_matrices, 1.0, (16, 16, 16), sidelength=8, batch_size=2
    )
    assert recon_no_batch.shape == (16, 16, 16)
    assert recon_with_batch.shape == (16, 16, 16)
    diff = torch.abs(recon_no_batch - recon_with_batch.to(recon_no_batch.device)).max()
    assert diff == 0.0
