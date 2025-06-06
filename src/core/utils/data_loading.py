"""Module for loading and processing Eclipse reservoir simulation output files."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import numpy as np
import numpy.typing as npt
import torch
from resdata import grid, resfile, summary, rft, well
import dask.array as da
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class EclipseDataLoader:
    """Class for loading and processing Eclipse simulation output files.
    
    This class handles loading of Eclipse simulation output files (.EGRID, .INIT, .UNRST, .SMSPEC)
    and provides methods to extract relevant data for PINO training.
    
    Attributes:
        grid_path: Path to the .EGRID file
        init_path: Path to the .INIT file
        restart_path: Path to the .UNRST file
        summary_path: Path to the .SMSPEC file
        grid: Grid object containing the reservoir grid
        init: ResdataInitFile object containing the initialization data
        restart: ResdataRestartFile object containing the restart data
        summary: Summary object containing the summary data
        chunk_size: Size of chunks for memory-efficient loading
        use_dask: Whether to use Dask for parallel processing
    """
    
    def __init__(
        self,
        grid_path: Union[str, Path],
        init_path: Optional[Union[str, Path]] = None,
        restart_path: Optional[Union[str, Path]] = None,
        summary_path: Optional[Union[str, Path]] = None,
        chunk_size: int = 1024,
        use_dask: bool = True,
    ) -> None:
        """Initialize the EclipseDataLoader.
        
        Args:
            grid_path: Path to the .EGRID file
            init_path: Optional path to the .INIT file
            restart_path: Optional path to the .UNRST file
            summary_path: Optional path to the .SMSPEC file
            chunk_size: Size of chunks for memory-efficient loading
            use_dask: Whether to use Dask for parallel processing
        """
        self.grid_path = Path(grid_path)
        self.init_path = Path(init_path) if init_path else None
        self.restart_path = Path(restart_path) if restart_path else None
        self.summary_path = Path(summary_path) if summary_path else None
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        
        # Load the grid file
        self.grid = grid.Grid(str(self.grid_path))
        
        # Load init file if provided
        self.init = resfile.ResdataInitFile(self.grid, str(self.init_path)) if self.init_path else None
        
        # Load restart file if provided
        self.restart = resfile.ResdataRestartFile(self.grid, str(self.restart_path)) if self.restart_path else None
        
        # Load summary file if provided
        self.summary = summary.Summary(str(self.summary_path)) if self.summary_path else None
        
        # Get grid dimensions
        self.nx, self.ny, self.nz = self.grid.get_dims()
        
    def _load_chunked_data(
        self, data: np.ndarray, chunk_size: Optional[int] = None
    ) -> npt.NDArray[np.float32]:
        """Load data in chunks to handle large arrays.
        
        Args:
            data: Input data array
            chunk_size: Optional chunk size override
            
        Returns:
            Processed data array
        """
        chunk_size = chunk_size or self.chunk_size
        if self.use_dask:
            # Convert to Dask array for memory-efficient processing
            darr = da.from_array(data, chunks=chunk_size)
            return darr.compute().astype(np.float32)
        else:
            # Process in chunks using numpy
            n_chunks = (data.size + chunk_size - 1) // chunk_size
            processed_chunks = []
            for i in range(n_chunks):
                start = i * chunk_size
                end = min((i + 1) * chunk_size, data.size)
                chunk = data[start:end].astype(np.float32)
                processed_chunks.append(chunk)
            return np.concatenate(processed_chunks)
    
    def get_porosity(self) -> npt.NDArray[np.float32]:
        """Extract porosity data from the grid file.
        
        Returns:
            Array of porosity values with shape (nx, ny, nz)
        """
        poro = self.grid.export_porosity()
        return self._load_chunked_data(poro)
    
    def get_permeability(self) -> npt.NDArray[np.float32]:
        """Extract permeability data from the grid file.
        
        Returns:
            Array of permeability values with shape (nx, ny, nz)
        """
        perm = self.grid.export_permx()
        return self._load_chunked_data(perm)
    
    def get_pressure(self, timestep: int) -> npt.NDArray[np.float32]:
        """Extract pressure data for a specific timestep.
        
        Args:
            timestep: Index of the timestep to extract
            
        Returns:
            Array of pressure values with shape (nx, ny, nz)
            
        Raises:
            ValueError: If restart file is not loaded or timestep is invalid
        """
        if self.restart is None:
            raise ValueError("Restart file not loaded")
            
        pressure = self.restart.iget_kw("PRESSURE", timestep)
        return self._load_chunked_data(pressure)
    
    def get_saturation(self, timestep: int) -> Dict[str, npt.NDArray[np.float32]]:
        """Extract saturation data for a specific timestep.
        
        Args:
            timestep: Index of the timestep to extract
            
        Returns:
            Dictionary containing arrays of water, oil, and gas saturations
            with shape (nx, ny, nz)
            
        Raises:
            ValueError: If restart file is not loaded or timestep is invalid
        """
        if self.restart is None:
            raise ValueError("Restart file not loaded")
            
        swat = self.restart.iget_kw("SWAT", timestep)
        soil = self.restart.iget_kw("SOIL", timestep)
        sgas = self.restart.iget_kw("SGAS", timestep)
        
        return {
            "water": self._load_chunked_data(swat),
            "oil": self._load_chunked_data(soil),
            "gas": self._load_chunked_data(sgas),
        }
    
    def get_well_data(self) -> Dict[str, List[float]]:
        """Extract well data from the summary file.
        
        Returns:
            Dictionary containing lists of well data (rates, pressures, etc.)
            
        Raises:
            ValueError: If summary file is not loaded
        """
        if self.summary is None:
            raise ValueError("Summary file not loaded")
            
        # Extract well data
        well_data = {}
        for key in self.summary.keys():
            if key.startswith(("WOPR", "WGPR", "WWPR", "WBHP")):
                well_data[key] = self.summary.numpy_vector(key)
                
        return well_data
    
    def to_torch_tensor(
        self, data: npt.NDArray[np.float32], device: Optional[str] = None
    ) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor.
        
        Args:
            data: Numpy array to convert
            device: Optional device to move tensor to (e.g., "cuda" or "cpu")
            
        Returns:
            PyTorch tensor
        """
        tensor = torch.from_numpy(data)
        if device:
            tensor = tensor.to(device)
        return tensor

class EnsembleDataLoader:
    """Class for loading and processing multiple Eclipse simulation outputs.
    
    This class handles loading of multiple Eclipse simulation outputs and provides
    methods to extract and process data for ensemble-based PINO training.
    
    Attributes:
        data_loaders: List of EclipseDataLoader instances
        max_workers: Maximum number of parallel workers for data loading
    """
    
    def __init__(
        self,
        ensemble_paths: List[Dict[str, Union[str, Path]]],
        max_workers: int = 4,
        chunk_size: int = 1024,
        use_dask: bool = True,
    ) -> None:
        """Initialize the EnsembleDataLoader.
        
        Args:
            ensemble_paths: List of dictionaries containing paths to simulation files
                          for each ensemble member
            max_workers: Maximum number of parallel workers for data loading
            chunk_size: Size of chunks for memory-efficient loading
            use_dask: Whether to use Dask for parallel processing
        """
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.use_dask = use_dask
        
        # Initialize data loaders
        self.data_loaders = [
            EclipseDataLoader(
                **paths,
                chunk_size=chunk_size,
                use_dask=use_dask
            ) for paths in ensemble_paths
        ]
    
    def _parallel_load(
        self,
        func: callable,
        *args,
        **kwargs
    ) -> List[Any]:
        """Load data in parallel using ThreadPoolExecutor.
        
        Args:
            func: Function to apply to each data loader
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            List of results from each data loader
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_loader = {
                executor.submit(func, loader, *args, **kwargs): loader
                for loader in self.data_loaders
            }
            for future in as_completed(future_to_loader):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error loading data: {e}")
                    raise
        return results
    
    def get_ensemble_porosity(self) -> npt.NDArray[np.float32]:
        """Extract porosity data from all ensemble members.
        
        Returns:
            Array of porosity values with shape (n_ensemble, nx, ny, nz)
        """
        results = self._parallel_load(lambda loader: loader.get_porosity())
        return np.stack(results)
    
    def get_ensemble_permeability(self) -> npt.NDArray[np.float32]:
        """Extract permeability data from all ensemble members.
        
        Returns:
            Array of permeability values with shape (n_ensemble, nx, ny, nz)
        """
        results = self._parallel_load(lambda loader: loader.get_permeability())
        return np.stack(results)
    
    def get_ensemble_pressure(self, timestep: int) -> npt.NDArray[np.float32]:
        """Extract pressure data from all ensemble members for a specific timestep.
        
        Args:
            timestep: Index of the timestep to extract
            
        Returns:
            Array of pressure values with shape (n_ensemble, nx, ny, nz)
        """
        results = self._parallel_load(
            lambda loader: loader.get_pressure(timestep)
        )
        return np.stack(results)
    
    def get_ensemble_saturation(self, timestep: int) -> Dict[str, npt.NDArray[np.float32]]:
        """Extract saturation data from all ensemble members for a specific timestep.
        
        Args:
            timestep: Index of the timestep to extract
            
        Returns:
            Dictionary containing arrays of water, oil, and gas saturations
            with shape (n_ensemble, nx, ny, nz)
        """
        results = self._parallel_load(
            lambda loader: loader.get_saturation(timestep)
        )
        
        saturations = {
            "water": [],
            "oil": [],
            "gas": [],
        }
        
        for result in results:
            for phase in saturations:
                saturations[phase].append(result[phase])
                
        return {
            phase: np.stack(values) for phase, values in saturations.items()
        }
    
    def get_ensemble_well_data(self) -> List[Dict[str, List[float]]]:
        """Extract well data from all ensemble members.
        
        Returns:
            List of dictionaries containing well data for each ensemble member
        """
        return self._parallel_load(lambda loader: loader.get_well_data())
    
    def save_ensemble_data(
        self,
        output_path: Union[str, Path],
        timesteps: Optional[List[int]] = None,
    ) -> None:
        """Save ensemble data to a pickle file.
        
        Args:
            output_path: Path to save the data
            timesteps: Optional list of timesteps to save
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get all data
        data = {
            "porosity": self.get_ensemble_porosity(),
            "permeability": self.get_ensemble_permeability(),
            "well_data": self.get_ensemble_well_data(),
        }
        
        # Add time-dependent data if timesteps provided
        if timesteps is not None:
            data["timesteps"] = timesteps
            data["pressure"] = np.stack([
                self.get_ensemble_pressure(t) for t in timesteps
            ])
            data["saturation"] = {
                phase: np.stack([
                    self.get_ensemble_saturation(t)[phase] for t in timesteps
                ]) for phase in ["water", "oil", "gas"]
            }
        
        # Save to pickle file
        with open(output_path, "wb") as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_ensemble_data(
        cls,
        input_path: Union[str, Path],
    ) -> Dict[str, Union[npt.NDArray[np.float32], List[Dict[str, List[float]]]]]:
        """Load ensemble data from a pickle file.
        
        Args:
            input_path: Path to the pickle file
            
        Returns:
            Dictionary containing the loaded data
        """
        input_path = Path(input_path)
        with open(input_path, "rb") as f:
            return pickle.load(f) 