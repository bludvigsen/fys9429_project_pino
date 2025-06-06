"""Data processing module for reservoir simulation data.

This module provides functionality for processing and preparing
reservoir simulation data for the PINO model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import resdata.summary

from .eclipse_reader import EclipseReader

import os
# Print logging info
print(f"Current directory: {os.getcwd()}")
print(f"Log file will be written to: {os.path.join(os.getcwd(), 'prepare_training_data.log')}")

# Ensure log directory exists and is writable
log_dir = "../../logs"  # Change this to your desired directory
os.makedirs(log_dir, exist_ok=True)  # Create directory if it doesn't exist

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(log_dir, 'prepare_training_data.log'))
    ]
)

class SimulationVariable:
    """Enumeration of simulation variables."""
    
    # Grid properties
    PERM = "PERM"  # Permeability
    PORO = "PORO"  # Porosity
    FAULT = "FAULT"  # Fault indicator
    
    # Initial conditions
    PINI = "PINI"  # Initial pressure
    SINI = "SINI"  # Initial saturation
    
    # Dynamic variables
    PRESSURE = "PRESSURE"  # Pressure
    SWAT = "SWAT"  # Water saturation
    SGAS = "SGAS"  # Gas saturation
    SOIL = "SOIL"  # Oil saturation
    
    @classmethod
    def get_all_variables(cls) -> List[str]:
        """Get list of all variable names.
        
        Returns:
            List of variable names
        """
        return [var for var in cls.__dict__.values() if isinstance(var, str)]

class SimulationDataProcessor:
    """Processor for reservoir simulation data."""
    
    def __init__(
        self,
        input_variables: List[str],
        output_variables: List[str]
    ) -> None:
        """Initialize the processor.
        
        Args:
            input_variables: List of input variable names
            output_variables: List of output variable names
        """
        self.input_variables = [str(var) for var in input_variables]
        self.output_variables = [str(var) for var in output_variables]
        logger.info(f"Initialized processor with {len(input_variables)} input and {len(output_variables)} output variables")
        
    def load_grid_data(self, filepath: Union[str, Path]) -> object:
        """Load grid data from EGRID file.
        
        Args:
            filepath: Path to EGRID file
            
        Returns:
            Grid object
        """
        logger.info(f"Loading grid data from: {filepath}")
        reader = EclipseReader(filepath)
        grid_obj = reader.read_grid()
        #print("JS FMU (106,164,191): ", grid_obj.nx, grid_obj.ny, grid_obj.nz)
        logger.info(f"Grid dimensions: {grid_obj.nx}x{grid_obj.ny}x{grid_obj.nz}")
        return grid_obj
        
    def load_restart_data(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load restart data from UNRST file.
        
        Args:
            filepath: Path to UNRST file
            
        Returns:
            Dictionary containing restart data
        """
        logger.info(f"Loading restart data from: {filepath}")
        reader = EclipseReader(filepath)
        data = reader.read_restart()
        logger.info(f"Loaded {len(data)} restart variables")
        return data
        
    def load_summary_data(self, filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
        """Load summary data from SMSPEC or UNSMRY file.
        
        Args:
            filepath: Path to SMSPEC or UNSMRY file
            
        Returns:
            Dictionary containing summary data
        """
        logger.info(f"Loading summary data from: {filepath}")
        
        try:
            # Create reader and pass the file path directly to read_summary
            reader = EclipseReader(filepath)
            df = reader.read_summary(filepath)
            
            # Convert DataFrame to dict of numpy arrays
            data = {}
            for column in df.columns:
                try:
                    data[column] = df[column].to_numpy().astype(np.float64)
                except (ValueError, TypeError):
                    logger.warning(f"Could not convert summary column {column} to numpy array")
            
            logger.info(f"Loaded summary data with {len(data)} columns and {len(df)} timesteps")
            return data
        except Exception as e:
            logger.error(f"Error loading summary data: {str(e)}")
            raise
        
    def load_ini_data(self, filepath: Union[str, Path]) -> Dict:
        """Load INI file data.
        
        Args:
            filepath: Path to INI file
            
        Returns:
            Dictionary containing INI parameters
        """
        logger.info(f"Loading INI data from: {filepath}")
        reader = EclipseReader(filepath)
        data = reader.read_init()
        logger.info(f"Loaded {len(data)} INI parameters")
        return data
        
    def process_simulation_data(
        self,
        grid_data: Optional[Dict[str, np.ndarray]] = None,
        restart_data: Optional[Dict[str, np.ndarray]] = None,
        summary_data: Optional[Dict[str, np.ndarray]] = None,
        ini_data: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Union[List[str], np.ndarray]]:
        """Process simulation data into input and output arrays.
        
        Args:
            grid_data: Grid data dictionary
            restart_data: Restart data dictionary
            summary_data: Summary data dictionary
            ini_data: INIT data dictionary
            
        Returns:
            Dictionary containing processed data arrays and variable names
        """
        try:
            # Initialize input and output data lists
            input_data = []
            output_data = []
            
            # Helper function to safely convert data to numpy array
            def to_numpy_array(data) -> Optional[np.ndarray]:
                if data is None:
                    return None
                if isinstance(data, np.ndarray):
                    return data.astype(np.float64)
                try:
                    return np.array(data, dtype=np.float64)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert data to numpy array: {str(e)}")
                    return None
            
            # Process grid data
            if grid_data:
                for var in self.input_variables:
                    if var in grid_data:
                        data = to_numpy_array(grid_data[var])
                        if data is not None:
                            input_data.append(data)
                        else:
                            logger.warning(f"Could not process grid data for {var}")
                            
                for var in self.output_variables:
                    if var in grid_data:
                        data = to_numpy_array(grid_data[var])
                        if data is not None:
                            output_data.append(data)
                        else:
                            logger.warning(f"Could not process grid data for {var}")
                        
            # Process restart data
            if restart_data:
                for var in self.input_variables:
                    if var in restart_data:
                        data = to_numpy_array(restart_data[var])
                        if data is not None:
                            input_data.append(data)
                        else:
                            logger.warning(f"Could not process restart data for {var}")
                            
                for var in self.output_variables:
                    if var in restart_data:
                        data = to_numpy_array(restart_data[var])
                        if data is not None:
                            output_data.append(data)
                        else:
                            logger.warning(f"Could not process restart data for {var}")
                        
            # Process summary data
            if summary_data:
                for var in self.input_variables:
                    if var in summary_data:
                        data = to_numpy_array(summary_data[var])
                        if data is not None:
                            input_data.append(data)
                        else:
                            logger.warning(f"Could not process summary data for {var}")
                            
                for var in self.output_variables:
                    if var in summary_data:
                        data = to_numpy_array(summary_data[var])
                        if data is not None:
                            output_data.append(data)
                        else:
                            logger.warning(f"Could not process summary data for {var}")
                        
            # Process INI data
            if ini_data:
                for var in self.input_variables:
                    if var in ini_data:
                        data = to_numpy_array(ini_data[var])
                        if data is not None:
                            input_data.append(data)
                        else:
                            logger.warning(f"Could not process INI data for {var}")
                            
                for var in self.output_variables:
                    if var in ini_data:
                        data = to_numpy_array(ini_data[var])
                        if data is not None:
                            output_data.append(data)
                        else:
                            logger.warning(f"Could not process INI data for {var}")
            
            # Stack arrays if we have any data
            if input_data:
                try:
                    input_array = np.stack(input_data)
                    logger.info(f"Stacked input data shape: {input_array.shape}")
                except ValueError as e:
                    logger.error(f"Failed to stack input arrays: {str(e)}")
                    raise
            else:
                logger.warning("No input data to process")
                input_array = np.array([])
                
            if output_data:
                try:
                    output_array = np.stack(output_data)
                    logger.info(f"Stacked output data shape: {output_array.shape}")
                except ValueError as e:
                    logger.error(f"Failed to stack output arrays: {str(e)}")
                    raise
            else:
                logger.warning("No output data to process")
                output_array = np.array([])
            
            return {
                "input_data": input_array,
                "output_data": output_array,
                "input_variables": self.input_variables,
                "output_variables": self.output_variables
            }
            
        except Exception as e:
            logger.error(f"Error processing simulation data: {str(e)}")
            raise

    def get_summary_vectors(self, filepath: Union[str, Path]) -> List[str]:
        """Get list of available vectors in a summary file without loading data.
        
        Args:
            filepath: Path to SMSPEC file
            
        Returns:
            List of vector names
        """
        logger.info(f"Getting vector list from: {filepath}")
        reader = EclipseReader(filepath)
        vectors = reader.read_summary_vectors(filepath)
        return vectors
        
    def load_selected_summary_data(self, filepath: Union[str, Path], selected_vectors: List[str]) -> Dict[str, np.ndarray]:
        """Load only selected vectors from summary file.
        
        Args:
            filepath: Path to SMSPEC file
            selected_vectors: List of vector names to load
            
        Returns:
            Dictionary containing only the selected summary vectors
        """
        logger.info(f"Loading selected summary vectors from: {filepath}")
        
        try:
            # Create reader and get case prefix
            reader = EclipseReader(filepath)
            
            # Get the file path without extension
            path = Path(filepath)
            if path.suffix.lower() in ['.smspec', '.sms', '.unsmry', '.usy']:
                case_prefix = str(path.with_suffix(''))
            else:
                case_prefix = str(path)
                
            # Create a Summary object
            summary_obj = resdata.summary.Summary(case_prefix)
            
            # Only extract the selected vectors
            data = {}
            
            # Always include TIME and DAYS if available
            time_vectors = ['TIME', 'DAYS', 'YEAR', 'MONTH', 'DAY']
            all_vectors = summary_obj.keys()
            
            # Add time vectors first if they exist
            for time_vec in time_vectors:
                if time_vec in all_vectors:
                    try:
                        data[time_vec] = summary_obj.numpy_vector(time_vec).astype(np.float64)
                    except Exception as e:
                        logger.warning(f"Failed to extract time vector {time_vec}: {str(e)}")
            
            # Add the selected vectors
            for vector in selected_vectors:
                if vector in all_vectors and vector not in data:  # Skip if already added as time vector
                    try:
                        data[vector] = summary_obj.numpy_vector(vector).astype(np.float64)
                    except Exception as e:
                        logger.warning(f"Failed to extract vector {vector}: {str(e)}")
            
            logger.info(f"Loaded {len(data)} summary vectors ({len(selected_vectors)} user-selected)")
            return data
            
        except Exception as e:
            logger.error(f"Error loading selected summary data: {str(e)}")
            raise

class ReservoirDataset(Dataset):
    """Dataset for reservoir simulation data."""
    
    def __init__(
        self,
        input_data: np.ndarray,
        output_data: np.ndarray
    ) -> None:
        """Initialize the dataset.
        
        Args:
            input_data: Input data array
            output_data: Output data array
        """
        self.input_data = torch.from_numpy(input_data).float()
        self.output_data = torch.from_numpy(output_data).float()
        logger.info(f"Created dataset with {len(self)} samples")
        
    def __len__(self) -> int:
        """Get dataset length.
        
        Returns:
            Number of samples
        """
        return len(self.input_data)
        
    def __getitem__(
        self,
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (input, output) tensors
        """
        return self.input_data[idx], self.output_data[idx] 