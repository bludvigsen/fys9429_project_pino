"""Main application for the PINO surrogate model.

This module provides the main entry point for the application,
including the GUI and model training pipeline.
"""

import sys
import os
import logging
from pathlib import Path

# Setup path for imports first
# Get the project root directory
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Set QT_QPA_PLATFORM=xcb for Linux if needed
if sys.platform.startswith('linux'):
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Disable QtWebEngine process sandbox (needed for network paths)
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
# Set additional QtWebEngine flags for network paths
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--no-sandbox --disable-gpu"

# Initialize Qt's WebEngine FIRST (before importing any PyQt modules)
import PyQt6.QtCore
PyQt6.QtCore.QCoreApplication.setAttribute(PyQt6.QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
# Use a try/except to handle different PyQt6 versions
try:
    # For newer PyQt6 versions
    PyQt6.QtCore.QCoreApplication.setAttribute(PyQt6.QtCore.Qt.ApplicationAttribute.AA_UseHighDpiPixmaps)
except AttributeError:
    try:
        # For older PyQt6 versions
        PyQt6.QtCore.QCoreApplication.setAttribute(PyQt6.QtCore.Qt.ApplicationAttribute.AA_EnableHighDpiScaling)
    except AttributeError:
        # Skip if neither attribute is available
        pass

from typing import Dict, List, Optional, Union
import torch
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QThread, pyqtSignal

from core.pino_model import PINOModel, PINOConfig, ReservoirDataset, PINOTrainer
from core.physics import PhysicsConfig, PhysicsLoss, DarcyLaw
from core.visualization import ReservoirVisualizer, ReservoirAnalyzer
from gui.main_window import MainWindow
from core.data_processing import SimulationDataProcessor
from core.eclipse_reader import EclipseReader
from app_controller import AppController
from core.logging_config import setup_logging

class TrainingThread(QThread):
    """Thread for model training."""
    
    progress = pyqtSignal(int)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        model: PINOModel,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        physics_config: PhysicsConfig,
        save_dir: Optional[Path],
        num_epochs: int
    ) -> None:
        """Initialize the training thread.
        
        Args:
            model: PINO model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            physics_config: Physics configuration
            save_dir: Optional directory to save checkpoints
            num_epochs: Number of epochs to train for
        """
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.physics_config = physics_config
        self.save_dir = save_dir
        self.num_epochs = num_epochs
        
    def run(self) -> None:
        """Run the training process."""
        try:
            # Initialize trainer
            trainer = PINOTrainer(
                model=self.model,
                train_loader=self.train_loader,
                val_loader=self.val_loader
            )
            
            # Initialize physics loss
            physics_loss = PhysicsLoss(self.physics_config)
            
            # Train model
            history = trainer.train(
                num_epochs=self.num_epochs,
                save_dir=self.save_dir
            )
            
            self.finished.emit(history)
            
        except Exception as e:
            self.error.emit(str(e))

def main():
    """Main entry point for the application."""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info("Starting PINO Surrogate application")
    
    # Create the application
    app = QApplication(sys.argv)
    app.setApplicationName("PINO Surrogate Model")
    
    # Create the main window
    controller = AppController()
    window = MainWindow(controller)
    window.setWindowTitle("PINO Surrogate Model")
    window.setMinimumSize(800, 600)
    window.show()
    
    # Run the application
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 