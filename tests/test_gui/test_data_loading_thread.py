"""Tests for the data loading thread."""

import pytest
from pathlib import Path
from PyQt6.QtCore import QObject
from PyQt6.QtTest import QSignalSpy

from src.gui.main_window import DataLoadingThread

@pytest.fixture
def sample_ensemble_files(tmp_path):
    """Create sample ensemble files for testing."""
    files = {
        "EGRID": {tmp_path / "test1.EGRID", tmp_path / "test2.EGRID"},
        "UNRST": {tmp_path / "test1.UNRST", tmp_path / "test2.UNRST"},
        "SMSPEC": {tmp_path / "test1.SMSPEC", tmp_path / "test2.SMSPEC"}
    }
    for file_set in files.values():
        for file_path in file_set:
            file_path.touch()
    return files

@pytest.fixture
def data_loading_thread(sample_ensemble_files):
    """Create a DataLoadingThread instance."""
    thread = DataLoadingThread(
        ensemble_files=sample_ensemble_files,
        chunk_size=1024,
        max_workers=4,
        use_dask=True
    )
    return thread

def test_thread_initialization(data_loading_thread):
    """Test initialization of DataLoadingThread."""
    assert data_loading_thread.chunk_size == 1024
    assert data_loading_thread.max_workers == 4
    assert data_loading_thread.use_dask is True
    assert data_loading_thread.isRunning() is False

def test_progress_signal(data_loading_thread):
    """Test progress signal emission."""
    spy = QSignalSpy(data_loading_thread.progress)
    data_loading_thread.progress.emit(50)
    assert spy.count() == 1
    assert spy[0][0] == 50

def test_completed_signal(data_loading_thread):
    """Test completed signal emission."""
    spy = QSignalSpy(data_loading_thread.completed)
    data_loading_thread.completed.emit()
    assert spy.count() == 1

def test_error_signal(data_loading_thread):
    """Test error signal emission."""
    spy = QSignalSpy(data_loading_thread.error)
    error_msg = "Test error"
    data_loading_thread.error.emit(error_msg)
    assert spy.count() == 1
    assert spy[0][0] == error_msg

def test_run_with_empty_files(tmp_path):
    """Test running thread with empty file sets."""
    empty_files = {
        "EGRID": set(),
        "UNRST": set(),
        "SMSPEC": set()
    }
    thread = DataLoadingThread(
        ensemble_files=empty_files,
        chunk_size=1024,
        max_workers=4,
        use_dask=True
    )
    spy = QSignalSpy(thread.error)
    thread.run()
    assert spy.count() == 1
    assert "No files" in spy[0][0]

def test_run_with_invalid_files(tmp_path):
    """Test running thread with invalid files."""
    invalid_files = {
        "EGRID": {tmp_path / "invalid.txt"},
        "UNRST": set(),
        "SMSPEC": set()
    }
    thread = DataLoadingThread(
        ensemble_files=invalid_files,
        chunk_size=1024,
        max_workers=4,
        use_dask=True
    )
    spy = QSignalSpy(thread.error)
    thread.run()
    assert spy.count() == 1
    assert "Invalid file" in spy[0][0]

def test_thread_cleanup(data_loading_thread):
    """Test thread cleanup."""
    data_loading_thread.start()
    data_loading_thread.wait()
    assert data_loading_thread.isRunning() is False 