"""Tests for the main window GUI."""

import pytest
from pathlib import Path
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow, FileDropWidget

@pytest.fixture
def app():
    """Create a QApplication instance."""
    app = QApplication([])
    yield app
    app.quit()

@pytest.fixture
def main_window(app):
    """Create a MainWindow instance."""
    window = MainWindow()
    window.show()
    yield window
    window.close()

@pytest.fixture
def sample_files(tmp_path):
    """Create sample Eclipse files for testing."""
    files = {}
    for ext in ["EGRID", "UNRST", "SMSPEC"]:
        file_path = tmp_path / f"test.{ext}"
        file_path.touch()
        files[ext] = file_path
    return files

def test_file_drop_widget_initialization(app):
    """Test initialization of FileDropWidget."""
    widget = FileDropWidget()
    assert widget.acceptDrops() is True
    assert widget.minimumHeight() == 200

def test_main_window_initialization(main_window):
    """Test initialization of MainWindow."""
    assert main_window.windowTitle() == "PINO Surrogate for Reservoir Simulation"
    assert main_window.minimumSize().width() == 800
    assert main_window.minimumSize().height() == 600

def test_handle_dropped_files(main_window, sample_files):
    """Test handling of dropped files."""
    # Simulate file drop
    for ext, file_path in sample_files.items():
        main_window.handle_dropped_files([file_path])
        
        # Check if file was added to ensemble_files
        assert file_path in main_window.ensemble_files[ext]
        
        # Check if load button was enabled
        assert main_window.load_button.isEnabled() is True

def test_update_file_list(main_window, sample_files):
    """Test updating the file list display."""
    # Add files
    for ext, file_path in sample_files.items():
        main_window.ensemble_files[ext].add(file_path)
    
    # Update file list
    main_window.update_file_list()
    
    # Check if all files are listed
    text = main_window.file_list.text()
    for ext, file_path in sample_files.items():
        assert file_path.name in text
        assert ext in text

def test_load_button_initial_state(main_window):
    """Test initial state of load button."""
    assert main_window.load_button.isEnabled() is False

def test_save_button_initial_state(main_window):
    """Test initial state of save button."""
    assert main_window.save_button.isEnabled() is False

def test_progress_bar_initial_state(main_window):
    """Test initial state of progress bar."""
    assert main_window.progress_bar.isVisible() is False

def test_chunk_size_control(main_window):
    """Test chunk size control."""
    assert main_window.chunk_size.value() == 1024
    assert main_window.chunk_size.minimum() == 64
    assert main_window.chunk_size.maximum() == 4096

def test_max_workers_control(main_window):
    """Test max workers control."""
    assert main_window.max_workers.value() == 4
    assert main_window.max_workers.minimum() == 1
    assert main_window.max_workers.maximum() == 16

def test_use_dask_checkbox(main_window):
    """Test use Dask checkbox."""
    assert main_window.use_dask.isChecked() is True

def test_status_bar_initial_state(main_window):
    """Test initial state of status bar."""
    assert main_window.statusBar().currentMessage() == "Ready" 