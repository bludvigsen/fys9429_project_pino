"""Main window for the PINO surrogate model GUI.

This module provides the main window interface for the PINO surrogate model application,
allowing users to load simulation files and configure model parameters.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog,
    QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QGroupBox, QFormLayout, QListWidget, QAbstractItemView,
    QLineEdit, QDialog, QSlider, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDropEvent
import pandas as pd

from core.data_processing import SimulationDataProcessor, SimulationVariable
from core.eclipse_reader import EclipseReader
from core.visualization import ReservoirVisualizer, ReservoirAnalyzer, PlotConfig
from core.pino_model import PINOModel, PINOConfig, ReservoirDataset, PINOTrainer
from core.physics import PhysicsConfig, PhysicsLoss, DarcyLaw

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('gui.log')  # Output to file
    ]
)
logger = logging.getLogger(__name__)

class SimulationLoaderThread(QThread):
    """Thread for loading simulation files."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        file_paths: Dict[str, Path],
        data_processor: SimulationDataProcessor
    ) -> None:
        """Initialize the loader thread.
        
        Args:
            file_paths: Dictionary mapping file types to paths
            data_processor: Data processor instance
        """
        super().__init__()
        self.file_paths = file_paths
        self.data_processor = data_processor
        logger.info(f"Initialized loader thread with {len(file_paths)} files")
    
    def run(self) -> None:
        """Run the loading process."""
        try:
            # First, ensure we have and load the grid file
            if "grid" not in self.file_paths:
                error_msg = "Grid file is required for reading restart and INIT files"
                logger.error(error_msg)
                raise ValueError(error_msg)
                
            logger.info("Loading grid data")
            # Get the grid object directly from load_grid_data
            grid_obj = self.data_processor.load_grid_data(
                self.file_paths["grid"]
            )
            
            if grid_obj is None:
                logger.error("Failed to get grid object")
                raise ValueError("Failed to get grid object")
            
            logger.info("Successfully retrieved grid object")
            
            # Now we can safely load other files
            restart_data = None
            if "restart" in self.file_paths:
                logger.info(f"Loading restart data from: {self.file_paths['restart']}")
                try:
                    # Create EclipseReader with the restart file path
                    eclipse_reader = EclipseReader(self.file_paths["restart"])
                    restart_data = eclipse_reader.read_restart(grid_obj)
                    logger.info(f"Successfully loaded restart data")
                except Exception as e:
                    logger.error(f"Error loading restart data: {str(e)}")
                    raise
            else:
                logger.warning("No restart file provided")
                
            # Load summary data
            summary_data = None
            if "summary" in self.file_paths:
                logger.info(f"Loading summary data from: {self.file_paths['summary']}")
                try:
                    # Load summary data using the path directly to handle both SMSPEC and UNSMRY
                    summary_data = self.data_processor.load_summary_data(
                        self.file_paths["summary"]
                    )
                    logger.info(f"Successfully loaded summary data")
                except Exception as e:
                    logger.error(f"Error loading summary data: {str(e)}")
                    raise
            else:
                logger.warning("No summary file provided")
                
            # Load INI data
            ini_data = None
            if "ini" in self.file_paths:
                logger.info(f"Loading INI data from: {self.file_paths['ini']}")
                try:
                    # Create EclipseReader with the INI file path
                    eclipse_reader = EclipseReader(self.file_paths["ini"])
                    ini_data = eclipse_reader.read_init(grid_obj)
                    logger.info(f"Successfully loaded INI data")
                except Exception as e:
                    logger.error(f"Error loading INI data: {str(e)}")
                    raise
            else:
                logger.warning("No INI file provided")
                
            # Extract grid data as dictionary for processing
            logger.info("Converting grid object to dictionary")
            grid_data = {}
            try:
                # Extract essential properties from grid object
                grid_data = {
                    'nx': grid_obj.nx,
                    'ny': grid_obj.ny,
                    'nz': grid_obj.nz
                }
                # Add more grid properties as needed
                logger.info(f"Extracted grid dimensions: {grid_data['nx']}x{grid_data['ny']}x{grid_data['nz']}")
            except Exception as e:
                logger.error(f"Error extracting grid data: {str(e)}")
                raise
                
            # Process data
            logger.info("Processing simulation data")
            try:
                processed_data = self.data_processor.process_simulation_data(
                    grid_data=grid_data,
                    restart_data=restart_data,
                    summary_data=summary_data,
                    ini_data=ini_data
                )
                logger.info("Data processing completed successfully")
                self.finished.emit(processed_data)
            except Exception as e:
                logger.error(f"Error in process_simulation_data: {str(e)}")
                raise
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            self.error.emit(str(e))

class DropZone(QLabel):
    """Custom label that accepts file drops."""
    
    file_dropped = pyqtSignal(str)
    
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        """Initialize the drop zone.
        
        Args:
            parent: Parent widget
        """
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
            }
            QLabel:hover {
                border-color: #666;
            }
        """)
        self.setText("Drop simulation files here\nor click to browse")
        self.setAcceptDrops(True)
        logger.debug("Initialized drop zone")
    
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter events.
        
        Args:
            event: Drag enter event
        """
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop events.
        
        Args:
            event: Drop event
        """
        files = [url.toLocalFile() for url in event.mimeData().urls()]
        if files:
            logger.info(f"File dropped: {files[0]}")
            self.file_dropped.emit(files[0])

class MainWindow(QMainWindow):
    """Main window for the PINO surrogate model application."""
    
    def __init__(self, app_controller):
        """Initialize the main window.
        
        Args:
            app_controller: Application controller
        """
        super().__init__()
        self.app_controller = app_controller
        self.logger = logging.getLogger(__name__)
        
        # Initialize UI components
        self.window = self
        self.menu_bar = self.menuBar()
        self.status_bar = self.statusBar()
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)
        self.setCentralWidget(self.central_widget)
        
        # Get data processor from app controller
        self.data_processor = self.app_controller.get_data_processor()
        
        # Cache for grid and property data
        self.cache = {
            'grid': None,
            'properties': {}
        }
        
        # Initialize simulation files dictionary
        self.simulation_files = {}
        
        # Initialize UI
        self.init_ui()
        
        # Set up logging
        self.setup_logging()
        
        # Initialize state
        self.processed_data: Optional[Dict] = None
        
        self.logger.info("Main window initialized")
        
    def init_ui(self) -> None:
        """Initialize the user interface."""
        # Set window title and size
        self.setWindowTitle("PINO Surrogate Model")
        self.setMinimumSize(800, 600)
        
        # File input section
        file_group = QGroupBox("Simulation Files")
        file_layout = QVBoxLayout()
        
        self.drop_zone = DropZone()
        self.drop_zone.file_dropped.connect(self.handle_file_drop)
        self.drop_zone.mousePressEvent = lambda _: self.browse_files()
        file_layout.addWidget(self.drop_zone)
        
        browse_button = QPushButton("Browse Files")
        browse_button.clicked.connect(self.browse_files)
        file_layout.addWidget(browse_button)
        
        file_group.setLayout(file_layout)
        self.layout.addWidget(file_group)
        
        # Model configuration section
        config_group = QGroupBox("Model Configuration")
        config_layout = QFormLayout()
        
        # Input variables
        input_layout = QVBoxLayout()
        input_layout.addWidget(QLabel("Input Variables:"))
        self.input_vars = QComboBox()
        self.input_vars.addItems([
            "PERM", "PORO", "FAULT", "PINI", "SINI",
            "PRESSURE", "SWAT", "SGAS", "SOIL"
        ])
        input_layout.addWidget(self.input_vars)
        
        # Output variables
        output_layout = QVBoxLayout()
        output_layout.addWidget(QLabel("Output Variables:"))
        self.output_vars = QComboBox()
        self.output_vars.addItems([
            "PRESSURE", "SWAT", "SGAS", "SOIL",
            "PERM", "PORO", "FAULT", "PINI", "SINI"
        ])
        output_layout.addWidget(self.output_vars)
        
        # Hidden dimensions
        self.hidden_dims = QSpinBox()
        self.hidden_dims.setRange(32, 512)
        self.hidden_dims.setValue(128)
        config_layout.addRow("Hidden Dimensions:", self.hidden_dims)
        
        # Number of layers
        self.num_layers = QSpinBox()
        self.num_layers.setRange(2, 8)
        self.num_layers.setValue(4)
        config_layout.addRow("Number of Layers:", self.num_layers)
        
        # Fourier modes
        self.fourier_modes = QSpinBox()
        self.fourier_modes.setRange(4, 32)
        self.fourier_modes.setValue(12)
        config_layout.addRow("Fourier Modes:", self.fourier_modes)
        
        # Dropout rate
        self.dropout_rate = QDoubleSpinBox()
        self.dropout_rate.setRange(0.0, 0.5)
        self.dropout_rate.setSingleStep(0.1)
        self.dropout_rate.setValue(0.1)
        config_layout.addRow("Dropout Rate:", self.dropout_rate)
        
        config_group.setLayout(config_layout)
        self.layout.addWidget(config_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.layout.addWidget(self.progress_bar)
        
        # Action buttons
        button_layout = QHBoxLayout()
        
        self.train_button = QPushButton("Train Model")
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setEnabled(False)
        button_layout.addWidget(self.train_button)
        
        self.visualize_button = QPushButton("Visualize Summary Results")
        self.visualize_button.clicked.connect(self.visualize_summary)
        button_layout.addWidget(self.visualize_button)
        
        # Add grid visualization button
        self.grid_button = QPushButton("Visualize Grid & Properties")
        self.grid_button.clicked.connect(self.visualize_grid)
        button_layout.addWidget(self.grid_button)
        
        self.layout.addLayout(button_layout)
        
        self.logger.info("UI setup completed")
        
    def visualize_summary(self) -> None:
        """Visualize summary data."""
        try:
            # Select summary file
            file_path = self.select_file("Select Eclipse Summary file", filter="Summary files (*.SMSPEC *.UNSMRY *.SMS *.USY)")
            if not file_path:
                return
            
            # Create a Path object for the summary file
            path = Path(file_path)
            
            # Preview the summary vectors
            self.preview_summary_vectors(path)
            
        except Exception as e:
            self.logger.error(f"Error visualizing summary data: {str(e)}", exc_info=True)
            self.show_error_dialog(f"Error visualizing summary data: {str(e)}")
            
    def visualize_results(self) -> None:
        """Redirect to visualize_summary for backward compatibility."""
        self.visualize_summary()
    
    def handle_file_drop(self, file_path: str) -> None:
        """Handle file drop events.
        
        Args:
            file_path: Path to dropped file
        """
        path = Path(file_path)
        file_type = path.suffix[1:].lower()  # Convert to lowercase for comparison
        
        # Map file extensions to internal types
        file_type_map = {
            'egrid': 'grid',
            'unrst': 'restart',
            'urs': 'restart',
            'ini': 'ini',
            'init': 'ini',
            'smspec': 'summary',
            'sms': 'summary',
            'unsmry': 'summary',
            'usy': 'summary'
        }
        
        if file_type not in file_type_map:
            logger.warning(f"Invalid file type: {path.suffix}")
            QMessageBox.warning(
                self,
                "Invalid File",
                "Please drop a valid simulation file (.egrid, .unrst/.urs, .ini/.init, .smspec/.sms, .unsmry/.usy)"
            )
            return
            
        internal_type = file_type_map[file_type]
        
        # Check if this type of file is already loaded
        if internal_type in self.simulation_files:
            existing_file = self.simulation_files[internal_type]
            # If it's the same file (or same base file for summary files), skip reloading
            if existing_file.stem == path.stem or existing_file == path:
                logger.info(f"File already loaded: {path.name}")
                QMessageBox.information(
                    self,
                    "File Already Loaded",
                    f"The file {path.name} (or a companion file) has already been loaded."
                )
                return
            else:
                # If it's a different file of the same type, confirm replacement
                response = QMessageBox.question(
                    self,
                    "Replace File?",
                    f"A {internal_type.upper()} file is already loaded. Replace it with {path.name}?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if response != QMessageBox.StandardButton.Yes:
                    return
        
        # Store the file in our dictionary
        self.simulation_files[internal_type] = path
        self.update_drop_zone_text()
        logger.info(f"Loaded file: {path.name}")
        
        # If it's a summary file, offer to visualize it immediately
        if internal_type == 'summary':
            self.offer_summary_visualization(path)
        
        # Only process data if we have all required files
        if self.has_required_files():
            self.load_simulation_data()
        else:
            self.update_drop_zone_text()
        
    def has_required_files(self) -> bool:
        """Check if all required files are loaded.
        
        Returns:
            bool: True if all required files are present
        """
        required_files = ['grid', 'restart', 'ini']  # Updated to match internal types
        return all(file_type in self.simulation_files for file_type in required_files)
        
    def update_drop_zone_text(self) -> None:
        """Update the drop zone text to show loaded files and missing requirements."""
        loaded_files = [f"{file_type.upper()}: {path.name}" 
                       for file_type, path in self.simulation_files.items()]
        
        if loaded_files:
            text = "Loaded files:\n" + "\n".join(loaded_files)
            
            # Show missing files
            missing_files = []
            if 'grid' not in self.simulation_files:
                missing_files.append("EGRID")
            if 'restart' not in self.simulation_files:
                missing_files.append("UNRST")
            if 'ini' not in self.simulation_files:
                missing_files.append("INIT")
                
            if missing_files:
                text += "\n\nMissing files:\n" + "\n".join(missing_files)
        else:
            text = "Drop simulation files here\nor click to browse"
            
        self.drop_zone.setText(text)
        
    def browse_files(self) -> None:
        """Open file browser dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Simulation File",
            "",
            "Simulation Files (*.egrid *.unrst *.urs *.ini *.init *.smry *.smspec *.sms *.unsmry *.usy);;All Files (*.*)"
        )
        if file_path:
            logger.info(f"Selected file: {file_path}")
            self.handle_file_drop(file_path)
            
    def load_simulation_data(self) -> None:
        """Load and process simulation data."""
        if not self.has_required_files():
            logger.warning("Cannot load data: missing required files")
            return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        # Create and start loader thread
        self.loader_thread = SimulationLoaderThread(
            self.simulation_files,
            self.data_processor
        )
        self.loader_thread.finished.connect(self.handle_loaded_data)
        self.loader_thread.error.connect(self.handle_load_error)
        self.loader_thread.start()
        
        logger.info("Started data loading thread")
        
    def handle_loaded_data(self, data: Dict) -> None:
        """Handle loaded simulation data.
        
        Args:
            data: Loaded and processed data
        """
        self.processed_data = data
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        self.visualize_button.setEnabled(True)
        self.grid_button.setEnabled(True)  # Enable grid button after data is loaded
        
        logger.info("Data loaded successfully")
        QMessageBox.information(
            self,
            "Success",
            "Simulation data loaded successfully"
        )
        
    def handle_load_error(self, error_msg: str) -> None:
        """Handle data loading errors.
        
        Args:
            error_msg: Error message
        """
        self.progress_bar.setVisible(False)
        logger.error(f"Data loading error: {error_msg}")
        QMessageBox.critical(
            self,
            "Error",
            f"Failed to load simulation data: {error_msg}"
        )
        
    def start_training(self) -> None:
        """Start model training."""
        if not self.processed_data:
            return
            
        # Create model configuration
        config = PINOConfig(
            input_dim=len(self.processed_data["input_variables"]),
            output_dim=len(self.processed_data["output_variables"]),
            hidden_dim=self.hidden_dims.value(),
            num_layers=self.num_layers.value(),
            fourier_modes=self.fourier_modes.value(),
            dropout_rate=self.dropout_rate.value()
        )
        
        # Create model
        model = PINOModel(config)
        
        # Create dataset
        dataset = ReservoirDataset(
            self.processed_data["input_data"],
            self.processed_data["output_data"]
        )
        
        # Create data loaders
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=32
        )
        
        # Create physics configuration
        physics_config = PhysicsConfig(
            enforce_conservation=True,
            enforce_boundaries=True,
            mass_weight=1.0,
            energy_weight=1.0,
            momentum_weight=1.0,
            pressure_weight=1.0,
            saturation_weight=1.0,
            porosity_weight=1.0,
            permeability_weight=1.0
        )
        
        # Create and start training thread
        self.training_thread = TrainingThread(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            physics_config=physics_config,
            save_dir=Path("checkpoints"),
            num_epochs=100
        )
        self.training_thread.progress.connect(self.update_training_progress)
        self.training_thread.finished.connect(self.handle_training_complete)
        self.training_thread.error.connect(self.handle_training_error)
        self.training_thread.start()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.train_button.setEnabled(False)
        
        logger.info("Started model training")
        
    def update_training_progress(self, epoch: int) -> None:
        """Update training progress bar.
        
        Args:
            epoch: Current epoch number
        """
        self.progress_bar.setValue(epoch)
        logger.debug(f"Training progress: {epoch}%")
        
    def handle_training_complete(self, history: Dict) -> None:
        """Handle training completion.
        
        Args:
            history: Training history
        """
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        
        logger.info("Model training completed successfully")
        QMessageBox.information(
            self,
            "Success",
            "Model training completed successfully"
        )
        
    def handle_training_error(self, error_msg: str) -> None:
        """Handle training errors.
        
        Args:
            error_msg: Error message
        """
        self.progress_bar.setVisible(False)
        self.train_button.setEnabled(True)
        
        logger.error(f"Training error: {error_msg}")
        QMessageBox.critical(
            self,
            "Error",
            f"Training failed: {error_msg}"
        )
        
    def setup_logging(self) -> None:
        """Set up logging configuration for the main window."""
        # Add a file handler for the main window
        try:
            file_handler = logging.FileHandler('main_window.log')
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(file_handler)
            self.logger.info("Main window logging configured")
        except Exception as e:
            print(f"Warning: Could not set up logging: {str(e)}")

    def offer_summary_visualization(self, path: Path) -> None:
        """Offer to visualize summary data immediately.
        
        Args:
            path: Path to summary file
        """
        response = QMessageBox.question(
            self,
            "Summary File Detected",
            f"Would you like to preview available vectors in {path.name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if response == QMessageBox.StandardButton.Yes:
            self.preview_summary_vectors(path)
    
    def preview_summary_vectors(self, path: Path) -> None:
        """Show available vectors before loading data.
        
        Args:
            path: Path to summary file
        """
        try:
            # Show loading progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Get vector list without loading data
            logger.info(f"Getting vector list from: {path}")
            vector_list = self.data_processor.get_summary_vectors(path)
            
            # Hide loading progress
            self.progress_bar.setVisible(False)
            
            # Show vector selection dialog
            selected_vectors = self.show_vector_selection_dialog(vector_list, path.name)
            
            if selected_vectors:
                # Load only the selected vectors
                self.load_selected_summary_data(path, selected_vectors)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            logger.error(f"Error getting summary vectors: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to get summary vectors: {str(e)}"
            )
    
    def show_vector_selection_dialog(self, vector_list: List[str], file_name: str) -> List[str]:
        """Show dialog for selecting vectors to load.
        
        Args:
            vector_list: List of available vectors
            file_name: Name of the summary file
            
        Returns:
            List of selected vector names
        """
        # Filter out time-related vectors which will be loaded automatically
        time_vectors = ['TIME', 'DAYS', 'YEAR', 'MONTH', 'DAY']
        data_vectors = [v for v in vector_list if v not in time_vectors]
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Select Vectors from {file_name}")
        dialog.setMinimumSize(500, 400)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Add info label
        info_label = QLabel(f"Found {len(data_vectors)} vectors in {file_name}. Select which ones to load:")
        layout.addWidget(info_label)
        
        # Add search/filter box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        filter_input = QLineEdit()
        search_layout.addWidget(filter_input)
        layout.addLayout(search_layout)
        
        # Add list widget for vector selection
        list_widget = QListWidget()
        list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        list_widget.addItems(data_vectors)
        layout.addWidget(list_widget)
        
        # Connect filter input to filter the list
        def filter_vectors(text):
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                item.setHidden(text.lower() not in item.text().lower())
        
        filter_input.textChanged.connect(filter_vectors)
        
        # Add select all / deselect all buttons
        select_layout = QHBoxLayout()
        
        select_all_button = QPushButton("Select All")
        select_all_button.clicked.connect(lambda: list_widget.selectAll())
        select_layout.addWidget(select_all_button)
        
        deselect_all_button = QPushButton("Deselect All")
        deselect_all_button.clicked.connect(lambda: list_widget.clearSelection())
        select_layout.addWidget(deselect_all_button)
        
        layout.addLayout(select_layout)
        
        # Add vector count display
        count_label = QLabel("0 vectors selected")
        layout.addWidget(count_label)
        
        # Update count when selection changes
        def update_count():
            count = len(list_widget.selectedItems())
            count_label.setText(f"{count} vectors selected")
            
        list_widget.itemSelectionChanged.connect(update_count)
        
        # Add buttons
        button_layout = QHBoxLayout()
        
        load_button = QPushButton("Load Selected")
        load_button.clicked.connect(dialog.accept)
        button_layout.addWidget(load_button)
        
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout)
        
        # Execute dialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_items = list_widget.selectedItems()
            return [item.text() for item in selected_items]
        else:
            return []
    
    def load_selected_summary_data(self, path: Path, selected_vectors: List[str]) -> None:
        """Load only selected summary vectors and visualize them.
        
        Args:
            path: Path to summary file
            selected_vectors: List of vectors to load
        """
        try:
            # Show loading progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)  # Indeterminate progress
            
            # Load only the selected vectors
            logger.info(f"Loading {len(selected_vectors)} vectors from: {path}")
            summary_data = self.data_processor.load_selected_summary_data(path, selected_vectors)
            
            # Convert to DataFrame
            df = pd.DataFrame(summary_data)
            
            # Hide loading progress
            self.progress_bar.setVisible(False)
            
            # Show visualization window with the selected data
            self.show_summary_visualizer(df, path.name)
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            logger.error(f"Error loading summary data: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to load summary data: {str(e)}"
            )
    
    def show_summary_visualizer(self, summary_data: pd.DataFrame, file_name: str) -> None:
        """Show the summary data visualization window.
        
        Args:
            summary_data: DataFrame containing summary data
            file_name: Name of the summary file
        """
        # Create a new window for summary visualization
        summary_window = QMainWindow(self)
        summary_window.setWindowTitle(f"Summary Data: {file_name}")
        summary_window.setMinimumSize(900, 700)
        
        # Create central widget
        central_widget = QWidget()
        summary_window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Add controls for selecting vectors to display
        controls_group = QGroupBox("Select Data to Display")
        controls_layout = QVBoxLayout()
        
        # Vector selection - use a list widget instead of combo box for multi-select
        vector_layout = QVBoxLayout()
        vector_layout.addWidget(QLabel("Available Vectors:"))
        
        # Create list widget for multi-selection
        self.vector_list_widget = QListWidget()
        self.vector_list_widget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        
        # Add all columns from the summary data
        time_vectors = ['TIME', 'DAYS', 'YEAR', 'MONTH', 'DAY']
        vector_columns = [col for col in summary_data.columns if col not in time_vectors]
        self.vector_list_widget.addItems(vector_columns)
        
        # Add search/filter box
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("Filter:"))
        self.vector_filter = QLineEdit()
        self.vector_filter.textChanged.connect(lambda text: self.filter_vectors(text, vector_columns))
        search_layout.addWidget(self.vector_filter)
        
        # Add filter to layout
        vector_layout.addLayout(search_layout)
        vector_layout.addWidget(self.vector_list_widget)
        controls_layout.addLayout(vector_layout)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        # Add plot button
        plot_button = QPushButton("Plot Selected Vectors")
        plot_button.clicked.connect(lambda: self.plot_summary_vectors(summary_data))
        button_layout.addWidget(plot_button)
        
        # Add stats button
        stats_button = QPushButton("Show Vector Statistics")
        stats_button.clicked.connect(lambda: self.show_vector_statistics(summary_data))
        button_layout.addWidget(stats_button)
        
        # Add export button
        export_button = QPushButton("Export Summary Data")
        export_button.clicked.connect(lambda: self.export_summary_data(summary_data, file_name))
        button_layout.addWidget(export_button)
        
        controls_layout.addLayout(button_layout)
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Add a placeholder for the plot
        self.plot_container = QLabel("Select vectors and click 'Plot Selected Vectors' to create a line plot")
        self.plot_container.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.plot_container.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
        layout.addWidget(self.plot_container)
        
        # Add the actual plot widget (will be populated later)
        self.plot_widget = None
        
        # Display summary statistics
        stats_group = QGroupBox("Summary Statistics")
        stats_layout = QVBoxLayout()
        
        # Calculate basic statistics
        num_vectors = len(vector_columns)
        num_steps = len(summary_data)
        
        stats_text = (
            f"Total vectors: {num_vectors}\n"
            f"Time steps: {num_steps}\n"
        )
        
        # If TIME or DAYS exists, add time range
        if 'TIME' in summary_data.columns:
            min_time = summary_data['TIME'].min()
            max_time = summary_data['TIME'].max()
            stats_text += f"Time range: {min_time:.2f} to {max_time:.2f} days\n"
        elif 'DAYS' in summary_data.columns:
            min_days = summary_data['DAYS'].min()
            max_days = summary_data['DAYS'].max()
            stats_text += f"Time range: {min_days:.2f} to {max_days:.2f} days\n"
        
        # Add YEAR range if available
        if 'YEAR' in summary_data.columns:
            min_year = summary_data['YEAR'].min()
            max_year = summary_data['YEAR'].max()
            stats_text += f"Year range: {min_year:.0f} to {max_year:.0f}\n"
            
        stats_label = QLabel(stats_text)
        stats_layout.addWidget(stats_label)
        
        # Add list of common vector prefixes to help users navigate
        vector_prefixes = set()
        for col in vector_columns:
            # Extract prefixes like WOPR, FOPT, etc.
            parts = col.split(':')
            if len(parts) > 0:
                vector_prefixes.add(parts[0])
        
        if vector_prefixes:
            prefix_text = "Common vector prefixes: " + ", ".join(sorted(list(vector_prefixes)[:15]))
            if len(vector_prefixes) > 15:
                prefix_text += f", ... and {len(vector_prefixes) - 15} more"
            prefix_label = QLabel(prefix_text)
            prefix_label.setWordWrap(True)
            stats_layout.addWidget(prefix_label)
        
        stats_group.setLayout(stats_layout)
        layout.addWidget(stats_group)
        
        # Store reference to window to prevent garbage collection
        self.summary_window = summary_window
        
        # Create a dictionary to store processed data
        self.processed_summary_data = self.preprocess_summary_data(summary_data)
        
        # Show the window
        summary_window.show()
    
    def filter_vectors(self, text: str, all_vectors: List[str]) -> None:
        """Filter vectors in the list widget based on search text.
        
        Args:
            text: Filter text
            all_vectors: List of all vector names
        """
        if not text:
            # If filter is empty, show all vectors
            self.vector_list_widget.clear()
            self.vector_list_widget.addItems(all_vectors)
            return
        
        # Filter vectors based on text
        filtered_vectors = [v for v in all_vectors if text.lower() in v.lower()]
        self.vector_list_widget.clear()
        self.vector_list_widget.addItems(filtered_vectors)
        
    def preprocess_summary_data(self, summary_data: pd.DataFrame) -> Dict:
        """Preprocess summary data for plotting.
        
        Args:
            summary_data: DataFrame containing summary data
            
        Returns:
            Dictionary containing processed data
        """
        processed_data = {}
        
        # Store original data
        processed_data['original_data'] = summary_data
        
        # Create sequential index for x-axis (ensures every timestep is plotted)
        processed_data['timestep_idx'] = np.arange(len(summary_data))
        
        # Store the actual YEAR values if available
        has_year_vector = 'YEAR' in summary_data.columns
        
        # Store YEAR vector directly if available
        if has_year_vector:
            processed_data['year_data'] = summary_data['YEAR'].values
            processed_data['x_label'] = 'Year'
            # No scaling needed for year data
            processed_data['use_year_directly'] = True
        else:
            processed_data['use_year_directly'] = False
            # Choose time data for axis labels when YEAR isn't available
            if 'TIME' in summary_data.columns:
                # TIME typically contains all timesteps in days
                processed_data['time_data'] = summary_data['TIME']
                processed_data['x_label'] = 'Time (years)'
                processed_data['x_scale'] = 365.25  # Days to years for display
            elif 'DAYS' in summary_data.columns:
                # DAYS contains all timesteps
                processed_data['time_data'] = summary_data['DAYS']
                processed_data['x_label'] = 'Time (years)'
                processed_data['x_scale'] = 365.25  # Days to years for display
            else:
                # Fall back to index
                processed_data['time_data'] = np.arange(len(summary_data))
                processed_data['x_label'] = 'Timestep'
                processed_data['x_scale'] = 1.0
            
        return processed_data
        
    def plot_summary_vectors(self, summary_data: pd.DataFrame) -> None:
        """Plot selected summary vectors as time series with all timesteps, displaying YEAR on x-axis.
        
        Args:
            summary_data: DataFrame containing summary data
        """
        if not hasattr(self, 'vector_list_widget'):
            return
            
        # Get selected vectors
        selected_items = self.vector_list_widget.selectedItems()
        selected_vectors = [item.text() for item in selected_items]
        
        if not selected_vectors:
            QMessageBox.information(
                self.summary_window,
                "No Selection",
                "Please select at least one vector to plot."
            )
            return
        
        # Get the processed data
        processed_data = self.processed_summary_data
        
        # Check if we have YEAR data to use directly
        use_year_directly = processed_data.get('use_year_directly', False)
        
        try:
            # Try to use matplotlib for plotting
            try:
                # Import plotting libraries
                import matplotlib
                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
                from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
                from matplotlib.figure import Figure
                import matplotlib.colors as mcolors
                import matplotlib.ticker as ticker
                
                # Create new figure with larger size for better visibility
                fig = Figure(figsize=(10, 7), dpi=100)
                
                # Create main subplot for the first vector
                ax = fig.add_subplot(111)
                
                # Use a color cycle for different vectors
                colors = list(mcolors.TABLEAU_COLORS.values())
                
                # Instead of using indices for x-axis, we'll use the actual YEAR values directly for plotting
                if use_year_directly:
                    year_data = processed_data['year_data']
                    x_label = 'Year'
                    
                    # Plot each selected vector with its own axis if we have multiple vectors
                    for i, vector in enumerate(selected_vectors):
                        if vector in summary_data.columns:
                            # Get y data for this vector
                            y_data = summary_data[vector]
                            
                            # First vector uses the main axis
                            if i == 0:
                                # Plot with a unique color, using YEAR directly for x-axis
                                line = ax.plot(year_data, y_data, color=colors[i % len(colors)], 
                                              linewidth=2, marker='.', markersize=3)
                                ax.set_ylabel(vector, color=colors[i % len(colors)])
                                ax.tick_params(axis='y', colors=colors[i % len(colors)])
                            else:
                                # Create a new axis for each additional vector
                                new_ax = ax.twinx()
                                
                                # If we have more than 2 vectors, offset the right axis
                                if i > 1:
                                    # Offset each additional axis
                                    offset = 60 * (i - 1)
                                    new_ax.spines['right'].set_position(('outward', offset))
                                    
                                # Plot on the new axis with a different color, using YEAR directly
                                line = new_ax.plot(year_data, y_data, color=colors[i % len(colors)], 
                                                  linewidth=2, marker='.', markersize=3)
                                new_ax.set_ylabel(vector, color=colors[i % len(colors)])
                                new_ax.tick_params(axis='y', colors=colors[i % len(colors)])
                                
                                # Store the axis for legend and other operations
                                if not hasattr(fig, 'twin_axes'):
                                    fig.twin_axes = []
                                fig.twin_axes.append(new_ax)
                    
                    # Set x-axis ticks to be at exact years with consistent spacing
                    min_year = int(np.floor(min(year_data)))
                    max_year = int(np.ceil(max(year_data)))
                    
                    # Calculate appropriate year step based on range
                    year_range = max_year - min_year
                    if year_range <= 10:
                        year_step = 1
                    elif year_range <= 20:
                        year_step = 2
                    elif year_range <= 50:
                        year_step = 5
                    else:
                        year_step = 10
                    
                    # Create array of years at regular intervals
                    tick_years = np.arange(min_year, max_year + 1, year_step)
                    
                    # Set exactly one tick per year at consistent positions
                    ax.set_xticks(tick_years)
                    ax.set_xticklabels([str(int(year)) for year in tick_years])
                    
                    # Ensure the axis limits cover the full data range
                    ax.set_xlim(min_year, max_year)
                    
                    # Use integer formatting for year labels
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                else:
                    # No direct year data, need to convert from time
                    time_data = processed_data['time_data']
                    x_scale = processed_data['x_scale']
                    x_label = processed_data['x_label']
                    
                    # Convert to years for plotting
                    years = time_data / x_scale
                    
                    for i, vector in enumerate(selected_vectors):
                        if vector in summary_data.columns:
                            # Get y data for this vector
                            y_data = summary_data[vector]
                            
                            # First vector uses the main axis
                            if i == 0:
                                # Plot with a unique color, using converted years for x-axis
                                line = ax.plot(years, y_data, color=colors[i % len(colors)], 
                                              linewidth=2, marker='.', markersize=3)
                                ax.set_ylabel(vector, color=colors[i % len(colors)])
                                ax.tick_params(axis='y', colors=colors[i % len(colors)])
                            else:
                                # Create a new axis for each additional vector
                                new_ax = ax.twinx()
                                
                                # If we have more than 2 vectors, offset the right axis
                                if i > 1:
                                    # Offset each additional axis
                                    offset = 60 * (i - 1)
                                    new_ax.spines['right'].set_position(('outward', offset))
                                    
                                # Plot on the new axis with a different color, using converted years
                                line = new_ax.plot(years, y_data, color=colors[i % len(colors)], 
                                                  linewidth=2, marker='.', markersize=3)
                                new_ax.set_ylabel(vector, color=colors[i % len(colors)])
                                new_ax.tick_params(axis='y', colors=colors[i % len(colors)])
                                
                                # Store the axis for legend and other operations
                                if not hasattr(fig, 'twin_axes'):
                                    fig.twin_axes = []
                                fig.twin_axes.append(new_ax)
                    
                    # Set x-axis ticks to be at exact years with consistent spacing
                    min_year = int(np.floor(min(years)))
                    max_year = int(np.ceil(max(years)))
                    
                    # Calculate appropriate year step based on range
                    year_range = max_year - min_year
                    if year_range <= 10:
                        year_step = 1
                    elif year_range <= 20:
                        year_step = 2
                    elif year_range <= 50:
                        year_step = 5
                    else:
                        year_step = 10
                    
                    # Create array of years at regular intervals
                    tick_years = np.arange(min_year, max_year + 1, year_step)
                    
                    # Set exactly one tick per year at consistent positions
                    ax.set_xticks(tick_years)
                    ax.set_xticklabels([str(int(year)) for year in tick_years])
                    
                    # Ensure the axis limits cover the full data range
                    ax.set_xlim(min(years), max(years))
                    
                    # Use integer formatting for year labels
                    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
                
                # Add labels and title
                ax.set_xlabel(x_label)
                
                if len(selected_vectors) == 1:
                    ax.set_title(f'{selected_vectors[0]} vs {x_label}')
                else:
                    ax.set_title(f'Multiple Vectors vs {x_label}')
                    
                # Add grid to the main axis
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Create a custom legend
                all_lines = []
                all_labels = []
                
                # Add lines from main axis
                for i, line in enumerate(ax.get_lines()):
                    all_lines.append(line)
                    all_labels.append(selected_vectors[i] if i < len(selected_vectors) else f"Line {i+1}")
                
                # Add lines from twin axes if they exist
                if hasattr(fig, 'twin_axes'):
                    for i, twin_ax in enumerate(fig.twin_axes):
                        for line in twin_ax.get_lines():
                            all_lines.append(line)
                            vector_index = i + 1  # +1 because main axis had first vector
                            if vector_index < len(selected_vectors):
                                all_labels.append(selected_vectors[vector_index])
                            else:
                                all_labels.append(f"Line {len(all_lines)}")
                
                # Add combined legend at the bottom of the plot
                fig.legend(all_lines, all_labels, loc='lower center', ncol=min(3, len(all_lines)), 
                          bbox_to_anchor=(0.5, 0.01))
                
                # Adjust layout to make room for the legend
                fig.subplots_adjust(bottom=0.15)
                
                # Create interactive canvas
                canvas = FigureCanvas(fig)
                canvas.setMinimumHeight(400)
                
                # Create navigation toolbar (zoom, pan, save, etc.)
                toolbar = NavigationToolbar(canvas, self.summary_window)
                
                # Create widget to hold the interactive plot
                if self.plot_widget is not None:
                    # Remove old widget
                    self.plot_widget.setParent(None)
                    
                plot_container = QWidget()
                plot_layout = QVBoxLayout(plot_container)
                
                # Add toolbar and canvas
                plot_layout.addWidget(toolbar)
                plot_layout.addWidget(canvas)
                
                # Add a note about the plot
                points_text = f"Showing all timesteps with evenly spaced year ticks"
                if len(selected_vectors) > 1:
                    note_text = f"{points_text}. Each vector is plotted on its own axis with its own scale. Use the toolbar above to zoom, pan, and export."
                else:
                    note_text = f"{points_text}. Use the toolbar above to zoom, pan, and export."
                note_label = QLabel(note_text)
                note_label.setStyleSheet("color: #555; font-style: italic;")
                note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                plot_layout.addWidget(note_label)
                
                # Replace placeholder with interactive plot
                layout = self.summary_window.centralWidget().layout()
                layout.replaceWidget(self.plot_container, plot_container)
                
                # Update references
                self.plot_container.deleteLater()
                self.plot_container = plot_container
                self.plot_widget = canvas
                
                # Store references to prevent garbage collection
                self.fig = fig
                self.toolbar = toolbar
                
                # Get the number of timesteps from the data
                num_timesteps = len(summary_data)
                logger.info(f"Created interactive plot with {len(selected_vectors)} vectors and {num_timesteps} timesteps")
                
            except ImportError as e:
                logger.warning(f"Interactive matplotlib not available: {str(e)}, falling back to static plot")
                self.plot_static_image(summary_data, selected_vectors, processed_data)
                
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            QMessageBox.warning(
                self.summary_window,
                "Plot Error",
                f"Failed to create plot: {str(e)}"
            )
            
    def plot_static_image(self, summary_data: pd.DataFrame, selected_vectors: List[str], processed_data: Dict) -> None:
        """Create a static image plot as fallback when interactive plot is not available.
        
        Args:
            summary_data: DataFrame containing summary data
            selected_vectors: List of selected vector names
            processed_data: Dictionary with processed plotting data
        """
        try:
            # Import plotting libraries
            import matplotlib
            matplotlib.use("agg")  # Non-interactive backend for figure creation
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            import matplotlib.colors as mcolors
            from matplotlib.ticker import FuncFormatter, MaxNLocator, MultipleLocator
            
            # Check if we have YEAR data to use directly
            use_year_directly = processed_data.get('use_year_directly', False)
            
            if use_year_directly:
                year_data = processed_data['year_data']
                x_label = 'Year'
                
                # Get min and max years
                min_year = np.min(year_data)
                max_year = np.max(year_data)
                # Round min year down and max year up
                min_year_tick = np.floor(min_year)
                max_year_tick = np.ceil(max_year)
                
                # Define index to year mapping function
                def get_index_for_year(year_value):
                    # Find closest year in the data
                    distances = np.abs(year_data - year_value)
                    closest_idx = np.argmin(distances)
                    return closest_idx
                
                # Create tick positions at yearly intervals
                tick_positions = []
                tick_labels = []
                for year in range(int(min_year_tick), int(max_year_tick) + 1):
                    idx = get_index_for_year(year)
                    if idx is not None:
                        tick_positions.append(idx)
                        tick_labels.append(str(int(year)))
            else:
                # Use time data for axis labels
                time_data = processed_data['time_data']
                x_label = processed_data['x_label']
                x_scale = processed_data['x_scale']
            
            # Create new figure with larger size for better visibility
            fig = Figure(figsize=(10, 7), dpi=100)
            
            # Create main subplot for the first vector
            ax = fig.add_subplot(111)
            
            # Define formatter function for x-axis
            if use_year_directly:
                # Use actual YEAR values
                def format_time_axis(x, pos):
                    idx = int(round(x))
                    if idx >= 0 and idx < len(year_data):
                        return f"{int(year_data[idx])}"
                    return ""
            else:
                # Convert from time vector
                def format_time_axis(x, pos):
                    idx = int(round(x))
                    if idx >= 0 and idx < len(time_data):
                        return f"{time_data[idx]/x_scale:.0f}"
                    return ""
            
            # Set custom formatter for x-axis
            ax.xaxis.set_major_formatter(FuncFormatter(format_time_axis))
            
            # If we have direct year data, set explicit ticks
            if use_year_directly and tick_positions:
                ax.set_xticks(tick_positions)
                ax.set_xticklabels(tick_labels)
            else:
                # Find indices that correspond to whole years
                if 'YEAR' in summary_data.columns:
                    year_series = summary_data['YEAR']
                    unique_years = sorted(year_series.unique())
                    year_indices = [year_series[year_series == year].index[0] for year in unique_years]
                    ax.set_xticks(year_indices)
                    ax.set_xticklabels([f"{int(year)}" for year in unique_years])
            
            # Use a color cycle for different vectors
            colors = list(mcolors.TABLEAU_COLORS.values())
            
            # Plot each selected vector with its own axis if we have multiple vectors
            for i, vector in enumerate(selected_vectors):
                if vector in summary_data.columns:
                    # Get y data for this vector
                    y_data = summary_data[vector]
                    
                    # First vector uses the main axis
                    if i == 0:
                        # Plot with a unique color
                        line = ax.plot(time_data, y_data, color=colors[i % len(colors)], 
                                      linewidth=2, marker='.', markersize=3)
                        ax.set_ylabel(vector, color=colors[i % len(colors)])
                        ax.tick_params(axis='y', colors=colors[i % len(colors)])
                    else:
                        # Create a new axis for each additional vector
                        new_ax = ax.twinx()
                        
                        # If we have more than 2 vectors, offset the right axis
                        if i > 1:
                            # Offset each additional axis
                            offset = 60 * (i - 1)
                            new_ax.spines['right'].set_position(('outward', offset))
                            
                        # Plot on the new axis with a different color
                        line = new_ax.plot(time_data, y_data, color=colors[i % len(colors)], 
                                          linewidth=2, marker='.', markersize=3)
                        new_ax.set_ylabel(vector, color=colors[i % len(colors)])
                        new_ax.tick_params(axis='y', colors=colors[i % len(colors)])
            
            # Add labels and title
            ax.set_xlabel(x_label)
            
            if len(selected_vectors) == 1:
                ax.set_title(f'{selected_vectors[0]} vs {x_label}')
            else:
                ax.set_title(f'Multiple Vectors vs {x_label}')
                
            # Add grid to the main axis
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Create a custom legend for all vectors
            handles = []
            labels = []
            for i, vector in enumerate(selected_vectors):
                # Create a line for the legend
                handle = plt.Line2D([0], [0], color=colors[i % len(colors)], linewidth=2)
                handles.append(handle)
                labels.append(vector)
            
            # Add legend at the bottom
            fig.legend(handles, labels, loc='lower center', ncol=min(3, len(selected_vectors)), 
                    bbox_to_anchor=(0.5, 0.01))
            
            # Adjust layout to make room for the legend
            fig.subplots_adjust(bottom=0.15)
            
            # Tight layout for the rest of the plot
            fig.tight_layout(rect=[0, 0.1, 1, 0.98])
            
            # Save figure to temporary file
            import tempfile
            import os
            temp_file = os.path.join(tempfile.gettempdir(), 'plot.png')
            fig.savefig(temp_file, dpi=100)
            plt.close(fig)
            
            # Create QLabel with the image
            from PyQt6.QtGui import QPixmap
            pixmap = QPixmap(temp_file)
            
            # Create image display widget
            image_label = QLabel()
            image_label.setPixmap(pixmap)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("border: 1px solid #ccc; padding: 5px;")
            
            # Create widget to hold the image
            if self.plot_widget is not None:
                # Remove old widget
                self.plot_widget.setParent(None)
                
            plot_container = QWidget()
            plot_layout = QVBoxLayout(plot_container)
            plot_layout.addWidget(image_label)
            
            # Add a note about the static nature
            note_label = QLabel(f"Note: Using static image mode. Showing all timesteps. Each vector is plotted on its own axis with its own scale.")
            note_label.setStyleSheet("color: #555; font-style: italic;")
            note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plot_layout.addWidget(note_label)
            
            # Replace placeholder with plot
            layout = self.summary_window.centralWidget().layout()
            layout.replaceWidget(self.plot_container, plot_container)
            
            # Update references
            self.plot_container.deleteLater()
            self.plot_container = plot_container
            self.plot_widget = image_label
            
            logger.info(f"Created static multi-axis line plot with {len(selected_vectors)} vectors and {len(time_data)} timesteps")
            
        except Exception as e:
            logger.error(f"Error creating static plot: {str(e)}")
            # Fall back to text display
            self.show_text_plot(summary_data, selected_vectors, processed_data)
    
    def show_text_plot(self, summary_data: pd.DataFrame, selected_vectors: List[str], processed_data: Dict) -> None:
        """Show a text-based representation of the plot.
        
        Args:
            summary_data: DataFrame containing summary data
            selected_vectors: List of selected vector names
            processed_data: Dictionary with processed plotting data
        """
        # Check if we have YEAR data to use directly
        use_year_directly = processed_data.get('use_year_directly', False)
        
        if use_year_directly:
            year_data = processed_data['year_data']
            x_label = 'Year'
            # Get start and end years
            start_time = year_data[0]
            end_time = year_data[-1]
        else:
            # Use time data for axis labels
            time_data = processed_data['time_data']
            x_label = processed_data['x_label']
            x_scale = processed_data['x_scale']
            # Calculate start and end times
            start_time = time_data[0] / x_scale
            end_time = time_data[-1] / x_scale
            
        # Create a text representation of the plot
        text = f"<html><body><h3>Plot of {len(selected_vectors)} vectors vs {x_label}</h3>"
        text += f"<p>Matplotlib not available. Showing data summary for {len(time_data)} timesteps.</p>"
        
        # Create a table with key points from each vector
        text += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
        text += "<tr><th>Vector</th><th>Start Time</th><th>Start Value</th><th>End Time</th><th>End Value</th><th>Min</th><th>Max</th></tr>"
        
        for vector in selected_vectors:
            if vector in summary_data.columns:
                y_data = summary_data[vector]
                
                start_value = y_data.iloc[0]
                end_value = y_data.iloc[-1]
                min_val = y_data.min()
                max_val = y_data.max()
                
                text += (
                    f"<tr>"
                    f"<td>{vector}</td>"
                    f"<td>{start_time:.2f}</td>"
                    f"<td>{start_value:.4g}</td>"
                    f"<td>{end_time:.2f}</td>"
                    f"<td>{end_value:.4g}</td>"
                    f"<td>{min_val:.4g}</td>"
                    f"<td>{max_val:.4g}</td>"
                    f"</tr>"
                )
        
        text += "</table>"
        
        # Add a note about exporting data for external plotting
        text += f"""
        <p style='margin-top: 20px;'>
        This summary represents {len(time_data)} timesteps. To create detailed plots, 
        you can export this data using the 'Export Summary Data' button
        and plot it with your preferred visualization tool.
        </p>
        """
        
        text += "</body></html>"
        
        # Create text display widget
        text_display = QLabel(text)
        text_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        text_display.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
        text_display.setWordWrap(True)
        
        # Replace the plot container with the text display
        if self.plot_widget is not None:
            # Remove old widget
            self.plot_widget.setParent(None)
            self.plot_widget = None
            
        # Replace placeholder with text display
        layout = self.summary_window.centralWidget().layout()
        layout.replaceWidget(self.plot_container, text_display)
        
        # Update references
        self.plot_container.deleteLater()
        self.plot_container = text_display
        
        logger.info(f"Created text-based plot representation for {len(selected_vectors)} vectors with {len(time_data)} timesteps")
    
    def show_vector_statistics(self, summary_data: pd.DataFrame) -> None:
        """Show statistics for selected summary vectors.
        
        Args:
            summary_data: DataFrame containing summary data
        """
        if not hasattr(self, 'vector_list_widget'):
            return
            
        # Get selected vectors
        selected_items = self.vector_list_widget.selectedItems()
        selected_vectors = [item.text() for item in selected_items]
        
        if not selected_vectors:
            QMessageBox.information(
                self.summary_window,
                "No Selection",
                "Please select at least one vector to view statistics."
            )
            return
            
        try:
            # Create a multi-vector statistics display
            stats_html = "<html><body>"
            stats_html += f"<h3>Statistics for {len(selected_vectors)} selected vectors</h3>"
            stats_html += "<table border='1' cellpadding='4' style='border-collapse: collapse;'>"
            stats_html += "<tr><th>Vector</th><th>Min</th><th>Max</th><th>Mean</th><th>Last Value</th></tr>"
            
            for vector in selected_vectors:
                if vector in summary_data.columns:
                    min_val = summary_data[vector].min()
                    max_val = summary_data[vector].max()
                    mean_val = summary_data[vector].mean()
                    last_val = summary_data[vector].iloc[-1]
                    
                    stats_html += (
                        f"<tr>"
                        f"<td>{vector}</td>"
                        f"<td>{min_val:.4g}</td>"
                        f"<td>{max_val:.4g}</td>"
                        f"<td>{mean_val:.4g}</td>"
                        f"<td>{last_val:.4g}</td>"
                        f"</tr>"
                    )
            
            stats_html += "</table>"
            stats_html += "</body></html>"
            
            # Create stats display widget
            stats_display = QLabel(stats_html)
            stats_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
            stats_display.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
            
            # Replace the plot container with the stats display
            if self.plot_widget is not None:
                # Remove old widget
                self.plot_widget.setParent(None)
                self.plot_widget = None
                
            # Replace placeholder with stats
            layout = self.summary_window.centralWidget().layout()
            layout.replaceWidget(self.plot_container, stats_display)
            
            # Update references
            self.plot_container.deleteLater()
            self.plot_container = stats_display
            
            logger.info(f"Displayed statistics for {len(selected_vectors)} summary vectors")
            
        except Exception as e:
            logger.error(f"Error displaying summary vector statistics: {str(e)}")
            QMessageBox.warning(
                self.summary_window,
                "Error",
                f"Failed to display statistics: {str(e)}"
            )
    
    def export_summary_data(self, summary_data: pd.DataFrame, file_name: str) -> None:
        """Export summary data to CSV file.
        
        Args:
            summary_data: DataFrame containing summary data
            file_name: Name of the source summary file
        """
        try:
            # Get save file path
            default_name = f"{Path(file_name).stem}_summary.csv"
            file_path, _ = QFileDialog.getSaveFileName(
            self,
                "Export Summary Data",
                default_name,
                "CSV Files (*.csv);;All Files (*.*)"
            )
            
            if not file_path:
                return  # User canceled
                
            # Export data to CSV
            summary_data.to_csv(file_path, index=False)
            
            logger.info(f"Exported summary data to {file_path}")
            QMessageBox.information(
                self.summary_window,
                "Export Successful",
                f"Summary data exported to {file_path}"
            )
            
        except Exception as e:
            logger.error(f"Error exporting summary data: {str(e)}")
            QMessageBox.warning(
                self.summary_window,
                "Export Error",
                f"Failed to export summary data: {str(e)}"
            )

    def visualize_grid(self) -> None:
        """Open a window to visualize grid, restart, and init files in 2D and 3D."""
        if not self.has_required_files():
            QMessageBox.warning(
                self,
                "Missing Files",
                "Grid visualization requires grid file (EGRID) and at least one of INIT or UNRST files."
            )
            return
        
        # Show progress while loading
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        
        try:
            # Create visualization window
            grid_window = QMainWindow(self)
            grid_window.setWindowTitle("Grid and Property Visualization")
            grid_window.setMinimumSize(1000, 800)
            
            # Create central widget
            central_widget = QWidget()
            grid_window.setCentralWidget(central_widget)
            layout = QVBoxLayout(central_widget)
            
            # Create visualizer instance
            from core.visualization import ReservoirVisualizer, PlotConfig
            visualizer = ReservoirVisualizer(PlotConfig(figure_size=(12, 10), dpi=100))
            
            # Load grid data first (required for all visualizations)
            grid_obj = None
            if 'grid' in self.simulation_files:
                try:
                    logger.info(f"Loading grid data for visualization from {self.simulation_files['grid']}")
                    grid_obj = self.data_processor.load_grid_data(self.simulation_files['grid'])
                    if grid_obj is None:
                        raise ValueError("Failed to load grid data")
                    
                    logger.info(f"Successfully loaded grid with dimensions {grid_obj.getNX()}x{grid_obj.getNY()}x{grid_obj.getNZ()}")
                except Exception as e:
                    logger.error(f"Error loading grid data: {str(e)}")
                    raise
            else:
                raise ValueError("Grid file (EGRID) is required for visualization")
            
            # Add visualization controls
            controls_group = QGroupBox("Visualization Controls")
            controls_layout = QFormLayout()
            
            # View type selection
            view_type = QComboBox()
            view_type.addItems(["2D View", "3D View"])
            controls_layout.addRow("View Type:", view_type)
            
            # Property selection
            property_combo = QComboBox()
            available_properties = []
            
            # Add grid properties
            property_combo.addItem("Grid Structure")
            
            # Create a reader for accessing properties
            eclipse_reader = EclipseReader("")
            
            # Add restart properties if available
            restart_data = None
            if 'restart' in self.simulation_files:
                try:
                    logger.info(f"Loading restart properties from {self.simulation_files['restart']}")
                    restart_reader = EclipseReader(self.simulation_files['restart'])
                    restart_data = restart_reader.read_restart(grid_obj)
                    
                    # Add property names from restart data
                    if restart_data:
                        # Get property list using the new method
                        restart_properties = restart_reader.get_property_list(restart_data)
                        for prop in restart_properties:
                            property_combo.addItem(f"Restart: {prop}")
                            available_properties.append(("restart", prop))
                        logger.info(f"Added {len(restart_properties)} restart properties")
                except Exception as e:
                    logger.error(f"Error loading restart properties: {str(e)}")
                    QMessageBox.warning(
                        self,
                        "Restart Load Warning",
                        f"Could not load all restart properties: {str(e)}"
                    )
            
            # Add init properties if available
            init_data = None
            if 'ini' in self.simulation_files:
                try:
                    logger.info(f"Loading init properties from {self.simulation_files['ini']}")
                    init_reader = EclipseReader(self.simulation_files['ini'])
                    init_data = init_reader.read_init(grid_obj)
                    
                    # Add property names from init data
                    if init_data:
                        # Get property list using the new method
                        init_properties = init_reader.get_property_list(init_data)
                        for prop in init_properties:
                            property_combo.addItem(f"Init: {prop}")
                            available_properties.append(("init", prop))
                        logger.info(f"Added {len(init_properties)} init properties")
                except Exception as e:
                    logger.error(f"Error loading init properties: {str(e)}")
                    QMessageBox.warning(
                        self,
                        "Init Load Warning",
                        f"Could not load all init properties: {str(e)}"
                    )
            
            controls_layout.addRow("Property:", property_combo)
            
            # For 2D view: layer selection
            layer_slider = QSlider(Qt.Orientation.Horizontal)
            layer_slider.setMinimum(0)
            layer_slider.setMaximum(grid_obj.getNZ() - 1)
            layer_slider.setValue(0)
            layer_label = QLabel("Layer: 1")
            
            # Update label when slider changes
            layer_slider.valueChanged.connect(lambda v: layer_label.setText(f"Layer: {v+1}"))
            
            # Initially hide layer controls for 3D view
            layer_widget = QWidget()
            layer_layout = QHBoxLayout(layer_widget)
            layer_layout.addWidget(QLabel("K Layer:"))
            layer_layout.addWidget(layer_slider)
            layer_layout.addWidget(layer_label)
            
            # Only show layer selection for 2D view
            view_type.currentTextChanged.connect(lambda t: layer_widget.setVisible(t == "2D View"))
            
            controls_layout.addRow("", layer_widget)
            
            # For 3D view: sampling controls
            sampling_widget = QWidget()
            sampling_layout = QHBoxLayout(sampling_widget)
            
            sampling_label = QLabel("Cell Sampling:")
            sampling_layout.addWidget(sampling_label)
            
            sampling_slider = QSlider(Qt.Orientation.Horizontal)
            sampling_slider.setMinimum(10)
            sampling_slider.setMaximum(100)
            sampling_slider.setValue(50)
            sampling_layout.addWidget(sampling_slider)
            
            sampling_value = QLabel("50%")
            sampling_slider.valueChanged.connect(lambda v: sampling_value.setText(f"{v}%"))
            sampling_layout.addWidget(sampling_value)
            
            # Only show sampling controls for 3D view
            view_type.currentTextChanged.connect(lambda t: sampling_widget.setVisible(t == "3D View"))
            sampling_widget.setVisible(False)  # Initially hidden
            
            controls_layout.addRow("", sampling_widget)
            
            # Slicing controls for property visualization
            slice_widget = QWidget()
            slice_layout = QHBoxLayout(slice_widget)
            
            slice_direction = QComboBox()
            slice_direction.addItems(["I Slice", "J Slice", "K Slice (Layers)"])
            slice_layout.addWidget(slice_direction)
            
            slice_slider = QSlider(Qt.Orientation.Horizontal)
            slice_slider.setMinimum(0)
            slice_slider.setMaximum(grid_obj.getNZ() - 1)  # Default to K direction
            slice_slider.setValue(0)
            slice_layout.addWidget(slice_slider)
            
            slice_label = QLabel("1")
            slice_layout.addWidget(slice_label)
            
            # Update slice slider range when direction changes
            def update_slice_range():
                if slice_direction.currentText() == "I Slice":
                    slice_slider.setMinimum(0)
                    slice_slider.setMaximum(grid_obj.getNX() - 1)
                    slice_slider.setValue(grid_obj.getNX() // 2)
                elif slice_direction.currentText() == "J Slice":
                    slice_slider.setMinimum(0)
                    slice_slider.setMaximum(grid_obj.getNY() - 1)
                    slice_slider.setValue(grid_obj.getNY() // 2)
                else:  # K Slice
                    slice_slider.setMinimum(0)
                    slice_slider.setMaximum(grid_obj.getNZ() - 1)
                    slice_slider.setValue(0)
                slice_label.setText(str(slice_slider.value() + 1))
            
            slice_direction.currentTextChanged.connect(update_slice_range)
            slice_slider.valueChanged.connect(lambda v: slice_label.setText(str(v + 1)))
            
            # Only show slicing controls for property visualization
            property_combo.currentTextChanged.connect(lambda t: slice_widget.setVisible(t != "Grid Structure"))
            slice_widget.setVisible(False)  # Initially hidden
            
            controls_layout.addRow("Property Slice:", slice_widget)
            
            # Apply button to generate visualization
            apply_button = QPushButton("Apply Visualization")
            controls_layout.addRow("", apply_button)
            
            # Add property info button to show metadata
            info_button = QPushButton("View Property Info")
            info_button.clicked.connect(lambda: self.show_property_info(
                grid_obj, property_combo.currentText(), restart_data, init_data
            ))
            controls_layout.addRow("", info_button)
            
            controls_group.setLayout(controls_layout)
            layout.addWidget(controls_group)
            
            # Add placeholder for plot
            plot_placeholder = QLabel("Select visualization options and click 'Apply Visualization'")
            plot_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            plot_placeholder.setStyleSheet("border: 1px solid #ccc; padding: 20px;")
            plot_placeholder.setMinimumHeight(500)
            layout.addWidget(plot_placeholder)
            
            # Function to update the visualization
            def update_visualization():
                try:
                    # Clear the previous plot
                    if hasattr(self, 'grid_plot_widget') and self.grid_plot_widget:
                        layout.removeWidget(self.grid_plot_widget)
                        self.grid_plot_widget.deleteLater()
                        
                    # Get visualization parameters
                    is_3d = view_type.currentText() == "3D View"
                    property_text = property_combo.currentText()
                    layer = layer_slider.value()
                    sampling = sampling_slider.value() / 100.0  # Convert to fraction
                    
                    # Determine the property to visualize
                    property_data = None
                    property_name = ""
                    
                    if property_text != "Grid Structure":
                        # Extract property source and name
                        source, prop_name = property_text.split(": ", 1)
                        property_name = prop_name
                        
                        # Load the property data
                        if source.lower() == "restart":
                            restart_reader = EclipseReader(self.simulation_files['restart'])
                            # Get property using the new method
                            property_data = restart_reader.get_property(restart_data, prop_name)
                        elif source.lower() == "init":
                            init_reader = EclipseReader(self.simulation_files['ini'])
                            # Get property using the new method
                            property_data = init_reader.get_property(init_data, prop_name)
                        
                        # Handle property data dimensionality
                        if property_data is not None:
                            try:
                                # Check the dimensionality
                                ndim = property_data.ndim
                                nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
                                
                                # Case 1: 1D array (most common for properties per active cell)
                                if ndim == 1:
                                    # Check the length
                                    if len(property_data) == nx * ny * nz:
                                        # This is likely a property for all cells
                                        property_data = property_data.reshape((nx, ny, nz))
                                    else:
                                        # This is likely a property only for active cells - need to expand
                                        expanded_data = np.full((nx, ny, nz), np.nan)
                                        
                                        # Now assign values only to active cells
                                        active_cells = 0
                                        for i in range(nx):
                                            for j in range(ny):
                                                for k in range(nz):
                                                    # Check if cell is active using the EclipseReader
                                                    if eclipse_reader.is_cell_active(grid_obj, i, j, k):
                                                        if active_cells < len(property_data):
                                                            expanded_data[i, j, k] = property_data[active_cells]
                                                            active_cells += 1
                                        
                                        property_data = expanded_data
                                        
                                # Case 2: 2D array (could be a layer or slice)
                                elif ndim == 2:
                                    # Get the dimensions
                                    dim1, dim2 = property_data.shape
                                    
                                    # Check if this is likely an IJ layer
                                    if dim1 == nx and dim2 == ny:
                                        # Expand to 3D with same values for all layers
                                        expanded_data = np.zeros((nx, ny, nz))
                                        for k in range(nz):
                                            expanded_data[:, :, k] = property_data
                                        property_data = expanded_data
                                    # Check if this could be a cross-section
                                    elif dim1 == nx and dim2 == nz:
                                        # Expand to 3D with same values for all J
                                        expanded_data = np.zeros((nx, ny, nz))
                                        for j in range(ny):
                                            expanded_data[:, j, :] = property_data
                                        property_data = expanded_data
                                    elif dim1 == ny and dim2 == nz:
                                        # Expand to 3D with same values for all I
                                        expanded_data = np.zeros((nx, ny, nz))
                                        for i in range(nx):
                                            expanded_data[i, :, :] = property_data
                                        property_data = expanded_data
                                    else:
                                        # If dimensions don't match, create a special 2D visualizer
                                        logger.warning(f"2D property data with shape {property_data.shape} doesn't match grid dimensions. Using special 2D visualization.")
                                        
                                # Case 3: 3D array already
                                elif ndim == 3:
                                    # Check if dimensions match
                                    if property_data.shape != (nx, ny, nz):
                                        logger.warning(f"3D property data shape {property_data.shape} doesn't match grid dimensions {(nx, ny, nz)}. Reshaping may be incorrect.")
                                        # Try to reshape if the total size matches
                                        if property_data.size == nx * ny * nz:
                                            property_data = property_data.reshape((nx, ny, nz))
                                        else:
                                            # Create a default array
                                            property_data = np.zeros((nx, ny, nz))
                                
                                # Log the final shape
                                logger.info(f"Property {property_name} prepared for visualization with shape {property_data.shape}")
                                
                            except Exception as e:
                                logger.warning(f"Could not reshape property data: {str(e)}")
                                # For 2D data that can't be reshaped, we'll handle specially
                                # during visualization
                    
                    # Determine visualization type
                    if is_3d:
                        # 3D visualization
                        try:
                            # Try to use plotly for interactive 3D
                            import plotly.graph_objects as go
                            from plotly.offline import plot
                            
                            # Check if property data is 3D
                            if property_data is not None and property_data.ndim == 2:
                                # Special case for 2D property data - create a 2D heatmap instead
                                fig = go.Figure()
                                fig.add_trace(
                                    go.Heatmap(
                                        z=property_data.T,  # Transpose for correct orientation
                                        colorscale="Viridis",
                                        colorbar=dict(title=property_name),
                                        hoverongaps=False
                                    )
                                )
                                
                                # Update layout for 2D view
                                fig.update_layout(
                                    title=f"{property_name} - 2D Property",
                                    xaxis_title="I",
                                    yaxis_title="J",
                                    height=800,
                                    width=1000
                                )
                            else:
                                # Normal 3D visualization
                                fig = visualizer.visualize_grid_structure(
                                    grid_obj,
                                    property_data=property_data,
                                    property_name=property_name,
                                    view_mode="3D",
                                    return_plotly=True
                                )
                            
                            # Convert to HTML and display in QWebEngineView
                            from PyQt6.QtWebEngineWidgets import QWebEngineView
                            
                            html = plot(fig, include_plotlyjs=True, output_type='div')
                            
                            web_view = QWebEngineView()
                            web_view.setHtml(html)
                            web_view.setMinimumHeight(500)
                            
                            # Replace placeholder with the plot
                            layout.replaceWidget(plot_placeholder, web_view)
                            self.grid_plot_widget = web_view
                            plot_placeholder.setVisible(False)
                            
                        except (ImportError, ModuleNotFoundError) as e:
                            logger.warning(f"Falling back to matplotlib due to import error: {str(e)}")
                            # Fall back to matplotlib
                            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                            
                            # Check if property data is 2D
                            if property_data is not None and property_data.ndim == 2:
                                # Create a 2D heatmap for matplotlib
                                import matplotlib.pyplot as plt  # Make sure plt is imported here
                                fig = plt.figure(figsize=(10, 8), dpi=100)
                                ax = fig.add_subplot(111)
                                
                                # Create a heatmap - handle potentially odd dimensions
                                try:
                                    im = ax.imshow(
                                        property_data.T,  # Transpose for correct orientation
                                        cmap='viridis',
                                        origin='lower',
                                        aspect='auto'
                                    )
                                except Exception as e:
                                    logger.warning(f"Error displaying transposed data: {str(e)}, trying without transpose")
                                    # Try without transpose for oddly shaped arrays
                                    im = ax.imshow(
                                        property_data,
                                        cmap='viridis',
                                        origin='lower',
                                        aspect='auto'
                                    )
                                
                                # Add colorbar
                                plt.colorbar(im, ax=ax, label=property_name)
                                
                                # Set labels and title
                                ax.set_xlabel("Column Index")
                                ax.set_ylabel("Row Index")
                                ax.set_title(f"{property_name} - 2D Property (Dimensions: {property_data.shape[0]}×{property_data.shape[1]})")
                            else:
                                # Normal 3D visualization
                                fig = visualizer._visualize_grid_matplotlib(
                                    grid_obj,
                                    property_data=property_data,
                                    property_name=property_name,
                                    view_mode="3D"
                                )
                            
                            canvas = FigureCanvasQTAgg(fig)
                            toolbar = NavigationToolbar2QT(canvas, grid_window)
                            
                            plot_widget = QWidget()
                            plot_layout = QVBoxLayout(plot_widget)
                            plot_layout.addWidget(toolbar)
                            plot_layout.addWidget(canvas)
                            
                            # Replace placeholder with the plot
                            layout.replaceWidget(plot_placeholder, plot_widget)
                            self.grid_plot_widget = plot_widget
                            plot_placeholder.setVisible(False)
                    else:
                        # 2D visualization
                        if property_text != "Grid Structure" and property_data is not None:
                            # For property data, show property slice
                            try:
                                # Get slice direction
                                direction = slice_direction.currentText().split()[0]  # "I", "J", or "K"
                                slice_idx = slice_slider.value()
                                
                                # Handle 2D property data specially
                                if property_data.ndim == 2:
                                    # Try to use plotly
                                    try:
                                        import plotly.graph_objects as go
                                        from plotly.offline import plot
                                        
                                        # Create a direct 2D visualization
                                        fig = go.Figure()
                                        fig.add_trace(
                                            go.Heatmap(
                                                z=property_data.T,  # Transpose for correct orientation
                                                colorscale="Viridis",
                                                colorbar=dict(title=property_name),
                                                hoverongaps=False
                                            )
                                        )
                                        
                                        # Update layout for 2D view
                                        fig.update_layout(
                                            title=f"{property_name} - 2D Property",
                                            xaxis_title="I",
                                            yaxis_title="J",
                                            height=800,
                                            width=1000
                                        )
                                    
                                        # Convert to HTML and display in QWebEngineView
                                        try:
                                            from PyQt6.QtWebEngineWidgets import QWebEngineView
                                            import os
                                            
                                            # Make sure sandbox is disabled for network paths
                                            os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
                                            
                                            html = plot(fig, include_plotlyjs=True, output_type='div')
                                            
                                            web_view = QWebEngineView()
                                            web_view.setHtml(html)
                                            web_view.setMinimumHeight(500)
                                            
                                            # Replace placeholder with the plot
                                            layout.replaceWidget(plot_placeholder, web_view)
                                            self.grid_plot_widget = web_view
                                            plot_placeholder.setVisible(False)
                                        except Exception as e:
                                            logger.warning(f"QtWebEngine error: {e}. Falling back to static plot.")
                                            # Fall back to matplotlib static image
                                            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                                            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                                            import matplotlib.pyplot as plt
                                            
                                            # Create a matplotlib figure
                                            fig_mpl = plt.figure(figsize=(10, 8), dpi=100)
                                            ax = fig_mpl.add_subplot(111)
                                            
                                            im = ax.imshow(
                                                property_data.T,  # Transpose for correct orientation
                                                cmap='viridis',
                                                origin='lower',
                                                aspect='auto'
                                            )
                                            
                                            plt.colorbar(im, ax=ax, label=property_name)
                                            ax.set_title(f"{property_name} - 2D Property")
                                            ax.set_xlabel("I")
                                            ax.set_ylabel("J")
                                            
                                            canvas = FigureCanvasQTAgg(fig_mpl)
                                            toolbar = NavigationToolbar2QT(canvas, grid_window)
                                            
                                            plot_widget = QWidget()
                                            plot_layout = QVBoxLayout(plot_widget)
                                            plot_layout.addWidget(toolbar)
                                            plot_layout.addWidget(canvas)
                                            
                                            # Replace placeholder with the plot
                                            layout.replaceWidget(plot_placeholder, plot_widget)
                                            self.grid_plot_widget = plot_widget
                                            plot_placeholder.setVisible(False)
                                    
                                    except (ImportError, ModuleNotFoundError) as e:
                                        logger.warning(f"Falling back to matplotlib due to import error: {str(e)}")
                                        # Fall back to matplotlib
                                        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                                        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                                        import matplotlib.pyplot as plt
                                        
                                        # Create a matplotlib figure
                                        fig = plt.figure(figsize=(10, 8), dpi=100)
                                        ax = fig.add_subplot(111)
                                        
                                        im = ax.imshow(
                                            property_data.T,  # Transpose for correct orientation
                                            cmap='viridis',
                                            origin='lower',
                                            aspect='auto'
                                        )
                                        
                                        plt.colorbar(im, ax=ax, label=property_name)
                                        ax.set_title(f"{property_name} - 2D Property")
                                        ax.set_xlabel("I")
                                        ax.set_ylabel("J")
                                        
                                        canvas = FigureCanvasQTAgg(fig)
                                        toolbar = NavigationToolbar2QT(canvas, grid_window)
                                        
                                        plot_widget = QWidget()
                                        plot_layout = QVBoxLayout(plot_widget)
                                        plot_layout.addWidget(toolbar)
                                        plot_layout.addWidget(canvas)
                                        
                                        # Replace placeholder with the plot
                                        layout.replaceWidget(plot_placeholder, plot_widget)
                                        self.grid_plot_widget = plot_widget
                                        plot_placeholder.setVisible(False)
                                else:
                                    # Normal 3D property slice
                                    try:
                                        # Use plotly for 3D property
                                        import plotly.graph_objects as go
                                        from plotly.offline import plot
                                        
                                        fig = visualizer.visualize_property_slice(
                                            grid_obj,
                                            property_data=property_data,
                                            property_name=property_name,
                                            slice_direction=direction,
                                            slice_index=slice_idx,
                                            return_plotly=True
                                        )
                                        
                                        # Convert to HTML and display in QWebEngineView
                                        try:
                                            from PyQt6.QtWebEngineWidgets import QWebEngineView
                                            import os
                                            
                                            # Make sure sandbox is disabled for network paths
                                            os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
                                            
                                            html = plot(fig, include_plotlyjs=True, output_type='div')
                                            
                                            web_view = QWebEngineView()
                                            web_view.setHtml(html)
                                            web_view.setMinimumHeight(500)
                                            
                                            # Replace placeholder with the plot
                                            layout.replaceWidget(plot_placeholder, web_view)
                                            self.grid_plot_widget = web_view
                                            plot_placeholder.setVisible(False)
                                        except Exception as e:
                                            logger.warning(f"QtWebEngine error: {e}. Falling back to matplotlib.")
                                            # Fall back to matplotlib
                                            from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                                            from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                                            
                                            # Use visualizer method for matplotlib
                                            fig = visualizer._visualize_property_slice_matplotlib(
                                                grid_obj,
                                                property_data=property_data,
                                                property_name=property_name,
                                                slice_direction=direction,
                                                slice_index=slice_idx
                                            )
                                            
                                            canvas = FigureCanvasQTAgg(fig)
                                            toolbar = NavigationToolbar2QT(canvas, grid_window)
                                            
                                            plot_widget = QWidget()
                                            plot_layout = QVBoxLayout(plot_widget)
                                            plot_layout.addWidget(toolbar)
                                            plot_layout.addWidget(canvas)
                                            
                                            # Replace placeholder with the plot
                                            layout.replaceWidget(plot_placeholder, plot_widget)
                                            self.grid_plot_widget = plot_widget
                                            plot_placeholder.setVisible(False)
                                    except (ImportError, ModuleNotFoundError) as e:
                                        logger.warning(f"Falling back to matplotlib due to import error: {str(e)}")
                                        # Fall back to matplotlib
                                        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                                        from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                                        
                                        # Use visualizer method for matplotlib
                                        fig = visualizer._visualize_property_slice_matplotlib(
                                            grid_obj,
                                            property_data=property_data,
                                            property_name=property_name,
                                            slice_direction=direction,
                                            slice_index=slice_idx
                                        )
                                        
                                        canvas = FigureCanvasQTAgg(fig)
                                        toolbar = NavigationToolbar2QT(canvas, grid_window)
                                        
                                        plot_widget = QWidget()
                                        plot_layout = QVBoxLayout(plot_widget)
                                        plot_layout.addWidget(toolbar)
                                        plot_layout.addWidget(canvas)
                                        
                                        # Replace placeholder with the plot
                                        layout.replaceWidget(plot_placeholder, plot_widget)
                                        self.grid_plot_widget = plot_widget
                                        plot_placeholder.setVisible(False)
                            except Exception as e:
                                logger.error(f"Error visualizing property: {str(e)}")
                                QMessageBox.warning(
                                    grid_window,
                                    "Property Visualization Error",
                                    f"Failed to visualize property: {str(e)}"
                                )
                        else:
                            # Show grid structure in 2D
                            try:
                                # Try to use plotly
                                import plotly.graph_objects as go
                                from plotly.offline import plot
                                
                                fig = visualizer.visualize_grid_structure(
                                    grid_obj,
                                    property_data=None,
                                    layer=layer,
                                    view_mode="2D",
                                    return_plotly=True
                                )
                                
                                # Convert to HTML and display in QWebEngineView
                                from PyQt6.QtWebEngineWidgets import QWebEngineView
                                
                                html = plot(fig, include_plotlyjs=True, output_type='div')
                                
                                web_view = QWebEngineView()
                                web_view.setHtml(html)
                                web_view.setMinimumHeight(500)
                                
                                # Replace placeholder with the plot
                                layout.replaceWidget(plot_placeholder, web_view)
                                self.grid_plot_widget = web_view
                                plot_placeholder.setVisible(False)
                                
                            except (ImportError, ModuleNotFoundError):
                                # Fall back to matplotlib
                                from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                                from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
                                
                                fig = visualizer._visualize_grid_matplotlib(
                                    grid_obj,
                                    property_data=None,
                                    layer=layer,
                                    view_mode="2D"
                                )
                                
                                canvas = FigureCanvasQTAgg(fig)
                                toolbar = NavigationToolbar2QT(canvas, grid_window)
                                
                                plot_widget = QWidget()
                                plot_layout = QVBoxLayout(plot_widget)
                                plot_layout.addWidget(toolbar)
                                plot_layout.addWidget(canvas)
                                
                                # Replace placeholder with the plot
                                layout.replaceWidget(plot_placeholder, plot_widget)
                                self.grid_plot_widget = plot_widget
                                plot_placeholder.setVisible(False)
                                
                except Exception as e:
                    logger.error(f"Error creating visualization: {str(e)}")
                    QMessageBox.warning(
                        grid_window,
                        "Visualization Error",
                        f"Failed to create visualization: {str(e)}"
                    )
            
            # Connect apply button to update function
            apply_button.clicked.connect(update_visualization)
            
            # Show the window
            self.grid_window = grid_window  # Store reference
            self.progress_bar.setVisible(False)
            grid_window.show()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            logger.error(f"Error opening grid visualization: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
            f"Failed to open grid visualization: {str(e)}"
            )

    def show_property_info(self, grid_obj, property_text: str, restart_data=None, init_data=None):
        """Show detailed property information to help diagnose dimensionality issues.
        
        Args:
            grid_obj: Grid object
            property_text: Property text from combo box (e.g. "Init: PORO")
            restart_data: Optional restart data object
            init_data: Optional init data object
        """
        try:
            if property_text == "Grid Structure":
                # Show grid information
                nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
                active_cells = 0
                
                # Try to get active cell count
                try:
                    if hasattr(grid_obj, 'getNumActive'):
                        active_cells = grid_obj.getNumActive()
                    else:
                        # Count manually using EclipseReader
                        eclipse_reader = EclipseReader("")
                        active_cells = eclipse_reader.count_active_cells(grid_obj)
                except Exception as e:
                    logger.warning(f"Could not count active cells: {str(e)}")
                
                info_text = (
                    f"<h3>Grid Information</h3>"
                    f"<p><b>Dimensions:</b> {nx} × {ny} × {nz}</p>"
                    f"<p><b>Total cells:</b> {nx * ny * nz}</p>"
                    f"<p><b>Active cells:</b> {active_cells}</p>"
                )
                
                # Try to get coordinate ranges
                try:
                    x_coords, y_coords, z_coords = [], [], []
                    
                    # Sample a few points to get coordinate ranges
                    for i in [0, nx//2, nx-1]:
                        for j in [0, ny//2, ny-1]:
                            for k in [0, nz//2, nz-1]:
                                try:
                                    # Get cell center
                                    if hasattr(grid_obj, 'getCellCenter'):
                                        x, y, z = grid_obj.getCellCenter(i, j, k)
                                        x_coords.append(x)
                                        y_coords.append(y)
                                        z_coords.append(z)
                                except:
                                    pass
                    
                    if x_coords and y_coords and z_coords:
                        info_text += (
                            f"<p><b>X range:</b> {min(x_coords):.2f} to {max(x_coords):.2f}</p>"
                            f"<p><b>Y range:</b> {min(y_coords):.2f} to {max(y_coords):.2f}</p>"
                            f"<p><b>Z range:</b> {min(z_coords):.2f} to {max(z_coords):.2f}</p>"
                        )
                except Exception as e:
                    logger.warning(f"Could not get coordinate ranges: {str(e)}")
                
                QMessageBox.information(
                    self,
                    "Grid Information",
                    info_text
                )
                return
            
            # Extract property source and name
            source, prop_name = property_text.split(": ", 1)
            
            # Get property data and metadata
            property_info = {}
            property_data = None
            
            if source.lower() == "restart" and restart_data:
                eclipse_reader = EclipseReader(self.simulation_files['restart'])
                property_info = eclipse_reader.get_property_info(restart_data, prop_name)
                try:
                    property_data = eclipse_reader.get_property(restart_data, prop_name)
                except Exception as e:
                    logger.warning(f"Could not load property data: {str(e)}")
            elif source.lower() == "init" and init_data:
                eclipse_reader = EclipseReader(self.simulation_files['ini'])
                property_info = eclipse_reader.get_property_info(init_data, prop_name)
                try:
                    property_data = eclipse_reader.get_property(init_data, prop_name)
                except Exception as e:
                    logger.warning(f"Could not load property data: {str(e)}")
            
            if not property_info.get("exists", False):
                QMessageBox.warning(
                    self,
                    "Property Not Found",
                    f"The property {prop_name} was not found in the {source} file."
                )
                return
            
            # Create info dialog content
            info_text = f"<h3>Property: {prop_name}</h3>"
            info_text += f"<p><b>Source:</b> {source}</p>"
            
            # Add property type if available
            if "property_type" in property_info:
                info_text += f"<p><b>Property type:</b> {property_info['property_type']}</p>"
                
            # Add shape information
            if "shape" in property_info:
                shape_str = " × ".join(str(dim) for dim in property_info["shape"])
                info_text += f"<p><b>Shape:</b> {shape_str}</p>"
                info_text += f"<p><b>Dimensions:</b> {property_info.get('ndim', '?')}</p>"
                
            # Add data type
            if "dtype" in property_info:
                info_text += f"<p><b>Data type:</b> {property_info['dtype']}</p>"
                
            # Add value range
            if "min" in property_info and "max" in property_info:
                info_text += f"<p><b>Value range:</b> {property_info['min']:.6g} to {property_info['max']:.6g}</p>"
                if "mean" in property_info:
                    info_text += f"<p><b>Mean value:</b> {property_info['mean']:.6g}</p>"
            
            # Add comparison with grid dimensions
            nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            active_cells = 0
            try:
                if hasattr(grid_obj, 'getNumActive'):
                    active_cells = grid_obj.getNumActive()
                else:
                    eclipse_reader = EclipseReader("")
                    active_cells = eclipse_reader.count_active_cells(grid_obj)
            except:
                pass
                
            info_text += f"<p><b>Grid dimensions:</b> {nx} × {ny} × {nz} = {nx * ny * nz} total cells</p>"
            info_text += f"<p><b>Active cells:</b> {active_cells}</p>"
            
            # Add histogram if property data is available
            if property_data is not None:
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
                    import numpy as np
                    
                    # Flatten the data and filter out NaN values
                    flat_data = property_data.flatten()
                    valid_data = flat_data[~np.isnan(flat_data)]
                    
                    if len(valid_data) > 0:
                        # Create histogram figure
                        fig = plt.figure(figsize=(6, 4))
                        ax = fig.add_subplot(111)
                        
                        # Create histogram with automatically determined bins
                        n_bins = min(100, max(10, int(np.sqrt(len(valid_data)))))
                        ax.hist(valid_data, bins=n_bins, alpha=0.7)
                        
                        ax.set_xlabel("Value")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"{prop_name} Distribution")
                        
                        plt.tight_layout()
                        
                        # Create canvas widget
                        canvas = FigureCanvasQTAgg(fig)
                        canvas.setMinimumHeight(300)
                        
                        # Create dialog with detailed info
                        dialog = QDialog(self)
                        dialog.setWindowTitle(f"Property Information: {prop_name}")
                        dialog.setMinimumWidth(600)
                        
                        layout = QVBoxLayout(dialog)
                        
                        # Add text information
                        info_label = QLabel(info_text)
                        info_label.setTextFormat(Qt.TextFormat.RichText)
                        info_label.setWordWrap(True)
                        layout.addWidget(info_label)
                        
                        # Add histogram
                        layout.addWidget(canvas)
                        
                        # Add close button
                        button = QPushButton("Close")
                        button.clicked.connect(dialog.close)
                        layout.addWidget(button)
                        
                        dialog.exec()
                        return
                except Exception as e:
                    logger.warning(f"Could not create histogram: {str(e)}")
            
            # If we can't create a histogram, just show the info text
            QMessageBox.information(
                self,
                f"Property Information: {prop_name}",
                info_text
            )
            
        except Exception as e:
            logger.error(f"Error showing property info: {str(e)}")
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to get property information: {str(e)}"
            )

    def visualize_init_property(self):
        """Visualize a property from an INIT file."""
        try:
            # Select INIT file
            file_path = self.select_file("Select Eclipse INIT file", filter="INIT files (*.INIT)")
            if not file_path:
                return
            
            # Check if we already loaded this grid
            if self.cache['grid'] is None or getattr(self.cache['grid'], '_file_path', None) != file_path:
                # Load grid from INIT file
                self.logger.info(f"Loading grid from INIT file: {file_path}")
                reader = EclipseReader(file_path)
                grid = reader.load_grid()
                
                # Store grid in cache with file path reference
                self.cache['grid'] = grid
                setattr(self.cache['grid'], '_file_path', file_path)
            else:
                self.logger.info(f"Using cached grid from: {getattr(self.cache['grid'], '_file_path', 'unknown')}")
                grid = self.cache['grid']
                reader = EclipseReader(file_path)
            
            # Get available properties
            self.properties = reader.get_available_properties()
            
            # Let user select a property
            property_name, ok = QInputDialog.getItem(
                self.window, 
                "Select Property", 
                "Choose property to visualize:", 
                self.properties, 
                0, 
                False
            )
            
            if not ok or not property_name:
                return
            
            # Check if property is cached
            if property_name in self.cache['properties']:
                self.logger.info(f"Using cached property: {property_name}")
                property_data = self.cache['properties'][property_name]
            else:
                # Get the property data
                self.logger.info(f"Loading property: {property_name}")
                property_data = reader.get_property(property_name)
                
                # Cache the property data
                self.cache['properties'][property_name] = property_data
            
            # Check if property data is available
            if property_data is None:
                self.show_error_dialog(f"Unable to retrieve property {property_name}")
                return
            
            # Visualize the property
            visualizer = ReservoirVisualizer()
            
            # Check for special handling based on property shape
            if property_data.ndim == 2 and (property_data.shape != (grid.getNX(), grid.getNY())):
                # Handle 2D property that doesn't match grid dimensions
                self.logger.warning(f"2D property data with shape {property_data.shape} doesn't match grid dimensions. Using special 2D visualization.")
            
            self.logger.info(f"Property {property_name} prepared for visualization with shape {property_data.shape} ")
            
            # Get slice direction and index from user if 3D property
            if property_data.ndim == 3 or (property_data.ndim == 2 and property_data.shape[0] == 1):
                # Ask for slice direction
                slice_direction, ok = QInputDialog.getItem(
                    self.window, 
                    "Select Slice Direction", 
                    "Choose slice direction:", 
                    ["I", "J", "K"], 
                    2,  # Default to K
                    False
                )
                
                if not ok:
                    return
                
                # Ask for slice index
                max_index = 0
                if slice_direction == "I":
                    max_index = grid.getNX() - 1
                elif slice_direction == "J":
                    max_index = grid.getNY() - 1
                else:  # K
                    max_index = grid.getNZ() - 1
                
                slice_index, ok = QInputDialog.getInt(
                    self.window,
                    "Select Slice Index",
                    f"Enter {slice_direction} slice index (0-{max_index}):",
                    0,  # Default value
                    0,  # Min value
                    max_index,  # Max value
                    1  # Step
                )
                
                if not ok:
                    return
                
                # Visualize the property slice
                visualizer.visualize_property_slice(
                    grid_obj=grid,
                    property_data=property_data,
                    property_name=property_name,
                    slice_direction=slice_direction,
                    slice_index=slice_index
                )
            else:
                # Simple 2D visualization
                visualizer.visualize_property_slice(
                    grid_obj=grid,
                    property_data=property_data,
                    property_name=property_name
                )
            
            self.status_bar.showMessage(f"Visualized property {property_name}", 5000)
            
        except Exception as e:
            self.logger.error(f"Error visualizing property: {str(e)}", exc_info=True)
            self.show_error_dialog(f"Error visualizing property: {str(e)}")

    # Show error dialog method
    def show_error_dialog(self, message: str) -> None:
        """Show an error dialog with the given message.
        
        Args:
            message: Error message to display
        """
        QMessageBox.critical(
            self.window,
            "Error",
            message
        )
    
    def select_file(self, title: str, filter: str = "All Files (*.*)") -> str:
        """Open a file dialog to select a file.
        
        Args:
            title: Dialog title
            filter: File filter string
            
        Returns:
            Selected file path or empty string if canceled
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self.window,
            title,
            "",
            filter
        )
        return file_path