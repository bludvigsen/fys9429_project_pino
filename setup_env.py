"""Environment setup script for the PINO surrogate model."""

import subprocess
import sys
import venv
from pathlib import Path

def setup_environment():
    """Set up the Python environment for the PINO surrogate model."""
    # Create virtual environment if it doesn't exist
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        venv.create("venv", with_pip=True)
    
    # Determine the pip path
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip"
    else:
        pip_path = venv_path / "bin" / "pip"
    
    # Upgrade pip
    print("Upgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"])
    
    # Install numpy first
    print("Installing numpy...")
    subprocess.run([str(pip_path), "install", "numpy>=1.24.0,<2.0.0"])
    
    # Install other requirements
    print("Installing other dependencies...")
    subprocess.run([str(pip_path), "install", "-r", "requirements.txt"])
    
    # Install the package in development mode
    print("Installing package in development mode...")
    subprocess.run([str(pip_path), "install", "-e", "."])
    
    print("\nSetup complete! To activate the environment:")
    if sys.platform == "win32":
        print("    .\\venv\\Scripts\\activate")
    else:
        print("    source venv/bin/activate")

if __name__ == "__main__":
    setup_environment() 