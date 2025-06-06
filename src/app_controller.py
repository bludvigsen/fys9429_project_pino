"""Application controller module."""
import logging
from typing import Dict, List, Any, Optional

from core.data_processing import SimulationDataProcessor
from core.data_processing import SimulationVariable

class AppController:
    """Controller for the PINO Surrogate application.
    
    This class manages the application state, data, and business logic.
    """
    
    def __init__(self):
        """Initialize the application controller."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing AppController")
        
        # Initialize data processor with default variables
        self.data_processor = SimulationDataProcessor(
            input_variables=[
                str(SimulationVariable.PERM),
                str(SimulationVariable.PORO),
                str(SimulationVariable.FAULT),
                str(SimulationVariable.PINI),
                str(SimulationVariable.SINI)
            ],
            output_variables=[
                str(SimulationVariable.PRESSURE),
                str(SimulationVariable.SWAT),
                str(SimulationVariable.SGAS),
                str(SimulationVariable.SOIL)
            ]
        )
        
        # Application state
        self.state = {
            'simulation_files': {},
            'current_project': None,
            'model_config': {
                'layers': 4,
                'features': 32,
                'learning_rate': 0.001,
                'batch_size': 16,
                'epochs': 100
            }
        }
        
    def get_simulation_files(self) -> Dict[str, str]:
        """Get the current simulation files.
        
        Returns:
            Dictionary mapping file types to file paths
        """
        return self.state['simulation_files']
    
    def set_simulation_file(self, file_type: str, file_path: str) -> None:
        """Set a simulation file.
        
        Args:
            file_type: Type of file (e.g., 'INIT', 'EGRID', 'UNRST')
            file_path: Path to the file
        """
        self.logger.info(f"Setting {file_type} file to {file_path}")
        self.state['simulation_files'][file_type] = file_path
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get the current model configuration.
        
        Returns:
            Dictionary with model configuration parameters
        """
        return self.state['model_config']
    
    def set_model_config(self, config: Dict[str, Any]) -> None:
        """Set the model configuration.
        
        Args:
            config: Dictionary with model configuration parameters
        """
        self.logger.info(f"Updating model configuration: {config}")
        self.state['model_config'].update(config)
    
    def get_data_processor(self) -> SimulationDataProcessor:
        """Get the data processor.
        
        Returns:
            SimulationDataProcessor instance
        """
        return self.data_processor 