"""Physics-Informed Neural Operator (PINO) model for reservoir simulation.

This module provides the PINO model implementation for reservoir simulation,
including the model architecture and training logic.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PINOConfig:
    """Configuration for the PINO model."""
    
    input_dim: int
    output_dim: int
    hidden_dim: int = 64
    num_layers: int = 4
    num_fourier_modes: int = 16
    num_fourier_layers: int = 4
    padding: int = 8
    dropout: float = 0.1
    activation: str = "gelu"
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.num_fourier_modes <= 0:
            raise ValueError("num_fourier_modes must be positive")
        if self.num_fourier_layers <= 0:
            raise ValueError("num_fourier_layers must be positive")
        if self.padding < 0:
            raise ValueError("padding must be non-negative")
        if not 0 <= self.dropout <= 1:
            raise ValueError("dropout must be between 0 and 1")
        if self.activation not in ["relu", "gelu", "tanh"]:
            raise ValueError("activation must be one of: relu, gelu, tanh")

class FourierLayer(nn.Module):
    """Fourier layer for the PINO model."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: int,
        activation: str = "gelu"
    ) -> None:
        """Initialize the Fourier layer.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            modes: Number of Fourier modes
            activation: Activation function to use
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize weights
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels, modes, modes, modes)
        )
        
        # Set activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Fourier layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, nx, ny, nz)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, nx, ny, nz)
        """
        batch_size = x.shape[0]
        
        # Compute Fourier transform
        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        
        # Apply weights in Fourier space
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.shape[-3],
            x.shape[-2] // 2 + 1,
            x.shape[-1] // 2 + 1,
            device=x.device,
            dtype=x.dtype
        )
        
        # Only use the first modes
        out_ft[..., :self.modes, :self.modes, :self.modes] = torch.einsum(
            "bixyz,ioxyz->boxyz",
            x_ft[..., :self.modes, :self.modes, :self.modes],
            self.weights
        )
        
        # Inverse Fourier transform
        x = torch.fft.irfftn(out_ft, dim=(-3, -2, -1))
        
        return self.activation(x)

class PINOModel(nn.Module):
    """Physics-Informed Neural Operator model."""
    
    def __init__(self, config: PINOConfig) -> None:
        """Initialize the PINO model.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        
        # Fourier layers
        self.fourier_layers = nn.ModuleList([
            FourierLayer(
                config.hidden_dim,
                config.hidden_dim,
                config.num_fourier_modes,
                config.activation
            ) for _ in range(config.num_fourier_layers)
        ])
        
        # MLP layers
        self.mlp_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.Dropout(config.dropout),
                nn.GELU() if config.activation == "gelu" else
                nn.ReLU() if config.activation == "relu" else
                nn.Tanh()
            ) for _ in range(config.num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the PINO model.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim, nx, ny, nz)
            
        Returns:
            Output tensor of shape (batch_size, output_dim, nx, ny, nz)
        """
        # Input projection
        x = self.input_proj(x.permute(0, 2, 3, 4, 1))
        x = x.permute(0, 4, 1, 2, 3)
        
        # Fourier layers
        for layer in self.fourier_layers:
            x = layer(x)
            
        # MLP layers
        x = x.permute(0, 2, 3, 4, 1)
        for layer in self.mlp_layers:
            x = layer(x)
            
        # Output projection
        x = self.output_proj(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x

class ReservoirDataset(Dataset):
    """Dataset for reservoir simulation data."""
    
    def __init__(
        self,
        data: Dict[str, npt.NDArray[np.float32]],
        input_keys: List[str],
        output_keys: List[str]
    ) -> None:
        """Initialize the dataset.
        
        Args:
            data: Dictionary containing simulation data
            input_keys: List of input variable names
            output_keys: List of output variable names
        """
        self.data = data
        self.input_keys = input_keys
        self.output_keys = output_keys
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self) -> None:
        """Validate the dataset."""
        # Check that all keys exist
        for key in self.input_keys + self.output_keys:
            if key not in self.data:
                raise ValueError(f"Missing data for key: {key}")
                
        # Check that all arrays have the same first dimension
        first_dim = next(iter(self.data.values())).shape[0]
        for key, value in self.data.items():
            if value.shape[0] != first_dim:
                raise ValueError(f"Inconsistent batch size for key: {key}")
                
    def __len__(self) -> int:
        """Get the number of samples in the dataset.
        
        Returns:
            Number of samples
        """
        return next(iter(self.data.values())).shape[0]
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (input tensor, output tensor)
        """
        # Get input data
        inputs = []
        for key in self.input_keys:
            inputs.append(self.data[key][idx])
        x = np.stack(inputs, axis=0)
        
        # Get output data
        outputs = []
        for key in self.output_keys:
            outputs.append(self.data[key][idx])
        y = np.stack(outputs, axis=0)
        
        return torch.from_numpy(x), torch.from_numpy(y)

class PINOTrainer:
    """Trainer for the PINO model."""
    
    def __init__(
        self,
        model: PINOModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0
    ) -> None:
        """Initialize the trainer.
        
        Args:
            model: PINO model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            device: Device to train on
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for optimization
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
    def train_epoch(self) -> float:
        """Train for one epoch.
        
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for x, y in self.train_loader:
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = self.criterion(y_pred, y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)
        
    def validate(self) -> float:
        """Validate the model.
        
        Returns:
            Average validation loss
            
        Raises:
            ValueError: If no validation loader is provided
        """
        if self.val_loader is None:
            raise ValueError("No validation loader provided")
            
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
        
    def train(
        self,
        num_epochs: int,
        save_dir: Optional[Union[str, Path]] = None,
        save_freq: int = 10
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train for
            save_dir: Optional directory to save checkpoints
            save_freq: Frequency of checkpoint saving
            
        Returns:
            Dictionary containing training history
        """
        history = {
            "train_loss": [],
            "val_loss": []
        }
        
        best_val_loss = float("inf")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch()
            history["train_loss"].append(train_loss)
            
            # Validate
            if self.val_loader is not None:
                val_loss = self.validate()
                history["val_loss"].append(val_loss)
                
                # Save best model
                if val_loss < best_val_loss and save_dir is not None:
                    best_val_loss = val_loss
                    self.save_checkpoint(save_dir / "best_model.pt")
                    
            # Save checkpoint
            if save_dir is not None and (epoch + 1) % save_freq == 0:
                self.save_checkpoint(save_dir / f"checkpoint_epoch_{epoch+1}.pt")
                
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.6f}")
            if self.val_loader is not None:
                print(f"Val Loss: {val_loss:.6f}")
                
        return history
        
    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.model.config
        }, path)
        
    def load_checkpoint(self, path: Union[str, Path]) -> None:
        """Load a model checkpoint.
        
        Args:
            path: Path to load the checkpoint from
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"]) 