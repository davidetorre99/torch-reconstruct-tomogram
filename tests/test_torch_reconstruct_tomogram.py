import torch_reconstruct_tomogram


def test_imports_with_version():
    assert isinstance(torch_reconstruct_tomogram.__version__, str)
