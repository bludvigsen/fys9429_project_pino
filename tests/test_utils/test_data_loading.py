"""Tests for the data loading module."""

import numpy as np
import pytest
from pathlib import Path
import torch

from src.core.utils.data_loading import EclipseDataLoader, EnsembleDataLoader

@pytest.fixture
def sample_eclipse_files(tmp_path):
    """Create sample Eclipse files for testing."""
    # Create sample .EGRID file
    grid_path = tmp_path / "test.EGRID"
    grid_path.touch()
    
    # Create sample .UNRST file
    restart_path = tmp_path / "test.UNRST"
    restart_path.touch()
    
    # Create sample .SMSPEC file
    summary_path = tmp_path / "test.SMSPEC"
    summary_path.touch()
    
    return {
        "grid_path": grid_path,
        "restart_path": restart_path,
        "summary_path": summary_path,
    }

def test_eclipse_data_loader_initialization(sample_eclipse_files):
    """Test initialization of EclipseDataLoader."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    
    assert loader.grid_path == sample_eclipse_files["grid_path"]
    assert loader.restart_path == sample_eclipse_files["restart_path"]
    assert loader.summary_path == sample_eclipse_files["summary_path"]

def test_ensemble_data_loader_initialization(sample_eclipse_files):
    """Test initialization of EnsembleDataLoader."""
    ensemble_paths = [sample_eclipse_files] * 3  # Create 3 ensemble members
    loader = EnsembleDataLoader(ensemble_paths)
    
    assert len(loader.data_loaders) == 3
    assert all(isinstance(loader, EclipseDataLoader) for loader in loader.data_loaders)

def test_get_porosity_shape(sample_eclipse_files):
    """Test shape of porosity data."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    poro = loader.get_porosity()
    
    assert isinstance(poro, np.ndarray)
    assert poro.dtype == np.float32
    assert len(poro.shape) == 3  # (nx, ny, nz)

def test_get_permeability_shape(sample_eclipse_files):
    """Test shape of permeability data."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    perm = loader.get_permeability()
    
    assert isinstance(perm, np.ndarray)
    assert perm.dtype == np.float32
    assert len(perm.shape) == 3  # (nx, ny, nz)

def test_get_pressure_shape(sample_eclipse_files):
    """Test shape of pressure data."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    pressure = loader.get_pressure(0)  # First timestep
    
    assert isinstance(pressure, np.ndarray)
    assert pressure.dtype == np.float32
    assert len(pressure.shape) == 3  # (nx, ny, nz)

def test_get_saturation_shape(sample_eclipse_files):
    """Test shape of saturation data."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    sat = loader.get_saturation(0)  # First timestep
    
    assert isinstance(sat, dict)
    assert all(phase in sat for phase in ["water", "oil", "gas"])
    assert all(isinstance(arr, np.ndarray) for arr in sat.values())
    assert all(arr.dtype == np.float32 for arr in sat.values())
    assert all(len(arr.shape) == 3 for arr in sat.values())  # (nx, ny, nz)

def test_ensemble_data_shapes(sample_eclipse_files):
    """Test shapes of ensemble data."""
    ensemble_paths = [sample_eclipse_files] * 3
    loader = EnsembleDataLoader(ensemble_paths)
    
    # Test porosity shape
    poro = loader.get_ensemble_porosity()
    assert poro.shape[0] == 3  # n_ensemble
    assert len(poro.shape) == 4  # (n_ensemble, nx, ny, nz)
    
    # Test permeability shape
    perm = loader.get_ensemble_permeability()
    assert perm.shape[0] == 3  # n_ensemble
    assert len(perm.shape) == 4  # (n_ensemble, nx, ny, nz)
    
    # Test pressure shape
    pressure = loader.get_ensemble_pressure(0)
    assert pressure.shape[0] == 3  # n_ensemble
    assert len(pressure.shape) == 4  # (n_ensemble, nx, ny, nz)
    
    # Test saturation shape
    sat = loader.get_ensemble_saturation(0)
    assert all(arr.shape[0] == 3 for arr in sat.values())  # n_ensemble
    assert all(len(arr.shape) == 4 for arr in sat.values())  # (n_ensemble, nx, ny, nz)

def test_to_torch_tensor(sample_eclipse_files):
    """Test conversion to PyTorch tensor."""
    loader = EclipseDataLoader(**sample_eclipse_files)
    data = np.random.rand(10, 10, 10).astype(np.float32)
    
    # Test CPU tensor
    tensor = loader.to_torch_tensor(data)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.dtype == torch.float32
    assert tensor.device.type == "cpu"
    
    # Test GPU tensor if available
    if torch.cuda.is_available():
        tensor = loader.to_torch_tensor(data, device="cuda")
        assert tensor.device.type == "cuda" 