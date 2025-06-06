"""Visualization and analysis tools for the PINO model.

This module provides tools for visualizing and analyzing the results
of the PINO model, including plotting functions and statistical analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sns
from dataclasses import dataclass
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

@dataclass
class PlotConfig:
    """Configuration for plotting."""
    
    # Figure settings
    figure_size: Tuple[float, float] = (10, 8)
    dpi: int = 100
    style: str = "default"
    
    # Color settings
    color_map: str = "viridis"
    
    # Layout settings
    show_grid: bool = True
    tight_layout: bool = True
    
    # Font settings
    font_size: int = 12
    title_font_size: int = 14
    label_font_size: int = 12
    tick_font_size: int = 10
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.figure_size[0] <= 0 or self.figure_size[1] <= 0:
            raise ValueError("Figure size must be positive")
        if self.dpi <= 0:
            raise ValueError("DPI must be positive")
        if self.font_size <= 0 or self.title_font_size <= 0 or self.label_font_size <= 0 or self.tick_font_size <= 0:
            raise ValueError("Font sizes must be positive")

class ReservoirVisualizer:
    """Visualization tools for reservoir simulation results."""
    
    def __init__(self, config: Optional[PlotConfig] = None) -> None:
        """Initialize the visualizer.
        
        Args:
            config: Optional plotting configuration
        """
        self.config = config or PlotConfig()
        
        # Set up matplotlib style
        plt.style.use(self.config.style)
        
        # Set up seaborn style
        sns.set_theme(style="whitegrid")
        
        # Set default font sizes
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['axes.titlesize'] = self.config.title_font_size
        plt.rcParams['axes.labelsize'] = self.config.label_font_size
        plt.rcParams['xtick.labelsize'] = self.config.tick_font_size
        plt.rcParams['ytick.labelsize'] = self.config.tick_font_size
        
        # Create custom colormap
        self.colormap = self._create_colormap()
        
    def _create_colormap(self) -> LinearSegmentedColormap:
        """Create a custom colormap for reservoir visualization.
        
        Returns:
            Custom colormap
        """
        colors = [
            (0.0, 0.0, 0.5),  # Dark blue
            (0.0, 0.5, 1.0),  # Light blue
            (0.5, 1.0, 0.5),  # Light green
            (1.0, 0.5, 0.0),  # Orange
            (1.0, 0.0, 0.0)   # Red
        ]
        return LinearSegmentedColormap.from_list('reservoir', colors)
        
    def plot_field(
        self,
        field: Union[np.ndarray, torch.Tensor],
        title: str,
        xlabel: str = "X",
        ylabel: str = "Y",
        zlabel: str = "Z",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Tuple[Figure, Axes]:
        """Plot a 3D field.
        
        Args:
            field: 3D field data
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            zlabel: Z-axis label
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            Tuple of (figure, axes)
        """
        # Convert to numpy if needed
        if isinstance(field, torch.Tensor):
            field = field.detach().cpu().numpy()
            
        # Create figure and axes
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot the field
        x, y, z = np.meshgrid(
            np.arange(field.shape[0]),
            np.arange(field.shape[1]),
            np.arange(field.shape[2])
        )
        
        scatter = ax.scatter(
            x, y, z,
            c=field.flatten(),
            cmap=self.colormap,
            alpha=0.6
        )
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Value')
        
        # Customize plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_zlabel(zlabel)
        ax.set_title(title)
        
        if self.config.show_grid:
            ax.grid(True)
            
        if self.config.tight_layout:
            plt.tight_layout()
            
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return fig, ax
        
    def plot_comparison(
        self,
        true_field: Union[np.ndarray, torch.Tensor],
        pred_field: Union[np.ndarray, torch.Tensor],
        title: str,
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Tuple[Figure, List[Axes]]:
        """Plot comparison between true and predicted fields.
        
        Args:
            true_field: True field data
            pred_field: Predicted field data
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            Tuple of (figure, list of axes)
        """
        # Convert to numpy if needed
        if isinstance(true_field, torch.Tensor):
            true_field = true_field.detach().cpu().numpy()
        if isinstance(pred_field, torch.Tensor):
            pred_field = pred_field.detach().cpu().numpy()
            
        # Create figure and axes
        fig = plt.figure(figsize=(3 * self.config.figure_size[0], self.config.figure_size[1]), dpi=self.config.dpi)
        
        # Plot true field
        ax1 = fig.add_subplot(131, projection="3d")
        x, y, z = np.meshgrid(
            np.arange(true_field.shape[0]),
            np.arange(true_field.shape[1]),
            np.arange(true_field.shape[2])
        )
        scatter1 = ax1.scatter(
            x, y, z,
            c=true_field.flatten(),
            cmap=self.colormap,
            alpha=0.6
        )
        ax1.set_title("True Field")
        plt.colorbar(scatter1, ax=ax1)
        
        # Plot predicted field
        ax2 = fig.add_subplot(132, projection="3d")
        scatter2 = ax2.scatter(
            x, y, z,
            c=pred_field.flatten(),
            cmap=self.colormap,
            alpha=0.6
        )
        ax2.set_title("Predicted Field")
        plt.colorbar(scatter2, ax=ax2)
        
        # Plot error
        ax3 = fig.add_subplot(133, projection="3d")
        error = np.abs(true_field - pred_field)
        scatter3 = ax3.scatter(
            x, y, z,
            c=error.flatten(),
            cmap="hot",
            alpha=0.6
        )
        ax3.set_title("Absolute Error")
        plt.colorbar(scatter3, ax=ax3)
        
        # Customize plots
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            if self.config.show_grid:
                ax.grid(True)
                
        if self.config.tight_layout:
            plt.tight_layout()
            
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return fig, [ax1, ax2, ax3]
        
    def plot_training_history(
        self,
        history: Dict[str, List[float]],
        title: str = "Training History",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Tuple[Figure, Axes]:
        """Plot training history.
        
        Args:
            history: Dictionary containing training history
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            Tuple of (figure, axes)
        """
        # Create figure and axes
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot each metric
        for metric, values in history.items():
            ax.plot(values, label=metric)
            
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        
        if self.config.show_grid:
            ax.grid(True)
            
        if self.config.tight_layout:
            plt.tight_layout()
            
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return fig, ax
        
    def plot_error_distribution(
        self,
        true_field: Union[np.ndarray, torch.Tensor],
        pred_field: Union[np.ndarray, torch.Tensor],
        title: str = "Error Distribution",
        save_path: Optional[Union[str, Path]] = None,
        show: bool = True
    ) -> Tuple[Figure, Axes]:
        """Plot error distribution.
        
        Args:
            true_field: True field data
            pred_field: Predicted field data
            title: Plot title
            save_path: Optional path to save the plot
            show: Whether to display the plot
            
        Returns:
            Tuple of (figure, axes)
        """
        # Convert to numpy if needed
        if isinstance(true_field, torch.Tensor):
            true_field = true_field.detach().cpu().numpy()
        if isinstance(pred_field, torch.Tensor):
            pred_field = pred_field.detach().cpu().numpy()
            
        # Compute error
        error = true_field - pred_field
        
        # Create figure and axes
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111)
        
        # Plot error distribution
        sns.histplot(error.flatten(), kde=True, ax=ax)
        
        # Customize plot
        ax.set_title(title)
        ax.set_xlabel("Error")
        ax.set_ylabel("Count")
        
        if self.config.show_grid:
            ax.grid(True)
            
        if self.config.tight_layout:
            plt.tight_layout()
            
        # Save plot if requested
        if save_path is not None:
            plt.savefig(save_path)
            
        if show:
            plt.show()
            
        return fig, ax

    def visualize_grid_structure(
        self,
        grid_obj,
        property_data: Optional[np.ndarray] = None,
        property_name: str = "",
        layer: int = 0,
        view_mode: str = "3D",
        return_plotly: bool = False
    ):
        """Visualize the grid structure with optional property data.
        
        Args:
            grid_obj: Eclipse grid object
            property_data: Optional property data to color the grid cells
            property_name: Name of the property data (for label)
            layer: Layer index for 2D view
            view_mode: Visualization mode ('2D' or '3D')
            return_plotly: Whether to return a plotly figure (for web)
            
        Returns:
            Plotly figure or matplotlib figure depending on return_plotly
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
            from plotly.subplots import make_subplots
            
            # Extract grid dimensions
            nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            
            # Create a figure
            if view_mode == "2D":
                fig = make_subplots(rows=1, cols=1)
                
                # Create a helper function to get cell corners
                def get_cell_corners(i, j, k):
                    # First check if cell is active before trying to get corners
                    try:
                        # Check if we have an EclipseReader instance with ACTNUM
                        from core.eclipse_reader import EclipseReader
                        reader = None
                        
                        # Try different methods to check if cell is active
                        is_active = False
                        try:
                            if hasattr(grid_obj, 'active'):
                                is_active = grid_obj.active(i, j, k)
                            elif hasattr(grid_obj, 'is_active'):
                                is_active = grid_obj.is_active(i, j, k)
                            else:
                                # Create a reader and use it to check
                                if reader is None:
                                    reader = EclipseReader("")
                                is_active = reader.is_cell_active(grid_obj, i, j, k)
                        except Exception:
                            # If we can't determine activity, assume it's active and try anyway
                            is_active = True
                        
                        # Skip inactive cells
                        if not is_active:
                            # Return zeros for inactive cells
                            print(f"Cell ({i},{j},{k}) is inactive, skipping corner calculation")
                            return np.zeros((4, 3))
                        
                        # Try different methods to get cell corners based on API
                        try:
                            # First try direct corner access
                            if hasattr(grid_obj, 'get_cell_corner'):
                                # New API
                                cell_corners = np.zeros((4, 3))
                                for corner_idx in range(4):
                                    x, y, z = grid_obj.get_cell_corner(corner_idx, i, j, k)
                                    cell_corners[corner_idx] = [x, y, z]
                                return cell_corners
                            elif hasattr(grid_obj, 'getCellCorner'):
                                # Old API
                                cell_corners = np.zeros((4, 3))
                                for corner_idx in range(4):
                                    # Pass ijk as a tuple without any parameter name
                                    x, y, z = grid_obj.getCellCorner(corner_idx, (i, j, k))
                                    cell_corners[corner_idx] = [x, y, z]
                                return cell_corners
                            elif hasattr(grid_obj, 'cell_corners'):
                                # Alternative API - returns all 8 corners, we take first 4
                                all_corners = grid_obj.cell_corners(i, j, k)
                                return all_corners[:4]
                            else:
                                # Try via EclipseReader
                                if reader is None:
                                    reader = EclipseReader("")
                                all_corners = reader.get_cell_corners(grid_obj, i, j, k)
                                return all_corners[:4]
                        except Exception as e:
                            print(f"Error getting corners for cell ({i},{j},{k}): {e}")
                            # Return zeros as fallback
                            return np.zeros((4, 3))
                    except Exception as e:
                        print(f"Error in get_cell_corners for cell ({i},{j},{k}): {e}")
                        # Return zeros as fallback
                        return np.zeros((4, 3))
                
                # Get corners for 2D view
                for i in range(nx):
                    for j in range(ny):
                        # Get the top face corners for this layer
                        corners = get_cell_corners(i, j, layer)
                
                        # Extract coordinates for plotting
                        x = [corners[k, 0] for k in range(4)] + [corners[0, 0]]
                        y = [corners[k, 1] for k in range(4)] + [corners[0, 1]]
                        
                        # Add color if property data is provided
                        if property_data is not None:
                            try:
                                color = property_data[i, j, layer]
                                fig.add_trace(
                                    go.Scatter(
                                        x=x, y=y,
                                        fill="toself",
                                        mode="lines",
                                        line=dict(color="black", width=0.5),
                                        marker=dict(color=color),
                                        showlegend=False
                                    )
                                )
                            except Exception as e:
                                # Fall back to uncolored cell
                                fig.add_trace(
                                    go.Scatter(
                                        x=x, y=y,
                                        fill="toself",
                                        mode="lines",
                                        line=dict(color="black", width=0.5),
                                        showlegend=False
                                    )
                                )
                        else:
                            fig.add_trace(
                                go.Scatter(
                                    x=x, y=y,
                                    fill="toself",
                                    mode="lines",
                                    line=dict(color="black", width=0.5),
                                    showlegend=False
                                )
                            )
                
                # Set axis labels
                fig.update_layout(
                    title=f"Grid Layer {layer+1}/{nz}" + (f" - {property_name}" if property_name else ""),
                    xaxis_title="X",
                    yaxis_title="Y",
                    height=800,
                    width=1000
                )
                
                # Add a colorbar if property data is provided
                if property_data is not None:
                    fig.update_layout(
                        coloraxis=dict(
                            colorscale="Viridis",
                            colorbar=dict(
                                title=property_name,
                                thickness=20,
                                len=0.75
                            )
                        )
                    )
                
            else:  # 3D mode
                # Create a 3D figure
                fig = go.Figure()
                
                # Create a helper function to get cell corners
                def get_cell_corners_3d(i, j, k):
                    # First check if cell is active before trying to get corners
                    try:
                        # Check if we have an EclipseReader instance with ACTNUM
                        from core.eclipse_reader import EclipseReader
                        reader = None
                        
                        # Try different methods to check if cell is active
                        is_active = False
                        try:
                            if hasattr(grid_obj, 'active'):
                                is_active = grid_obj.active(i, j, k)
                            elif hasattr(grid_obj, 'is_active'):
                                is_active = grid_obj.is_active(i, j, k)
                            else:
                                # Create a reader and use it to check
                                if reader is None:
                                    reader = EclipseReader("")
                                is_active = reader.is_cell_active(grid_obj, i, j, k)
                        except Exception:
                            # If we can't determine activity, assume it's active and try anyway
                            is_active = True
                            
                        # Skip inactive cells
                        if not is_active:
                            # Return zeros for inactive cells
                            print(f"Cell ({i},{j},{k}) is inactive, skipping corner calculation")
                            return np.zeros((8, 3))
                            
                        # Try different methods to get cell corners based on API
                        try:
                            # First try direct corner access
                            if hasattr(grid_obj, 'get_cell_corner'):
                                # New API
                                cell_corners = np.zeros((8, 3))
                                for corner_idx in range(8):
                                    x, y, z = grid_obj.get_cell_corner(corner_idx, i, j, k)
                                    cell_corners[corner_idx] = [x, y, z]
                                return cell_corners
                            elif hasattr(grid_obj, 'getCellCorner'):
                                # Old API
                                cell_corners = np.zeros((8, 3))
                                for corner_idx in range(8):
                                    # Pass ijk as a tuple without the parameter name
                                    x, y, z = grid_obj.getCellCorner(corner_idx, (i, j, k))
                                    cell_corners[corner_idx] = [x, y, z]
                                return cell_corners
                            elif hasattr(grid_obj, 'cell_corners'):
                                # Alternative API
                                return grid_obj.cell_corners(i, j, k)
                            else:
                                # Try via EclipseReader
                                if reader is None:
                                    reader = EclipseReader("")
                                return reader.get_cell_corners(grid_obj, i, j, k)
                        except Exception as e:
                            print(f"Error getting corners for cell ({i},{j},{k}): {e}")
                            # Return zeros as fallback
                            return np.zeros((8, 3))
                    except Exception as e:
                        print(f"Error in get_cell_corners_3d for cell ({i},{j},{k}): {e}")
                        # Return zeros as fallback
                        return np.zeros((8, 3))
                
                # If property data is provided, normalize it for color mapping
                if property_data is not None:
                    valid_data = property_data[~np.isnan(property_data)]
                    if len(valid_data) > 0:
                        min_val = np.nanmin(property_data)
                        max_val = np.nanmax(property_data)
                        if max_val > min_val:
                            norm_data = (property_data - min_val) / (max_val - min_val)
                        else:
                            norm_data = np.zeros_like(property_data)
                    else:
                        norm_data = np.zeros_like(property_data)
                
                # Sampling for visualization (plotting all cells can be too dense)
                skip_x = max(1, nx // 50)
                skip_y = max(1, ny // 50)
                skip_z = max(1, nz // 20)
                
                # For each grid cell
                for i in range(0, nx, skip_x):
                    for j in range(0, ny, skip_y):
                        for k in range(0, nz, skip_z):
                            # Skip inactive cells if we can detect them
                            try:
                                # Try different methods to check if cell is active
                                is_active = False
                                
                                if hasattr(grid_obj, 'active'):
                                    is_active = grid_obj.active(i, j, k)
                                elif hasattr(grid_obj, 'is_active'):
                                    is_active = grid_obj.is_active(i, j, k)
                                else:
                                    # Try via EclipseReader if we can't check directly
                                    from core.eclipse_reader import EclipseReader
                                    reader = EclipseReader("")
                                    is_active = reader.is_cell_active(grid_obj, i, j, k)
                                    
                                if not is_active:
                                    continue
                            except Exception as e:
                                # If we can't determine activity, continue anyway
                                print(f"Warning: Could not check if cell ({i},{j},{k}) is active: {e}")
                                
                            # Get cell corners (8 corners per cell)
                            try:
                                corners = get_cell_corners_3d(i, j, k)
                                
                                # Skip cells with all zero corners (inactive cells)
                                if np.allclose(corners, 0):
                                    continue
                                    
                                # Extract coordinates for each face of the cube
                                faces = [
                                    [0, 1, 2, 3],  # Bottom face
                                    [4, 5, 6, 7],  # Top face
                                    [0, 1, 5, 4],  # Front face
                                    [2, 3, 7, 6],  # Back face
                                    [0, 3, 7, 4],  # Left face
                                    [1, 2, 6, 5]   # Right face
                                ]
                                
                                # Define the color if property data is provided
                                if property_data is not None:
                                    try:
                                        color = norm_data[i, j, k]
                                        if np.isnan(color):
                                            color = 0.5  # Default gray for NaN values
                                    except:
                                        color = 0.5  # Default gray
                                else:
                                    color = 0.5  # Default gray
                                
                                # Add each face as a mesh3d
                                for face in faces:
                                    try:
                                        # Extract coordinates for this face
                                        x = [corners[idx, 0] for idx in face]
                                        y = [corners[idx, 1] for idx in face]
                                        z = [corners[idx, 2] for idx in face]
                                        
                                        # Skip zero or NaN coordinates
                                        if all(np.isclose(x, 0) and np.isclose(y, 0) and np.isclose(z, 0)):
                                            continue
                                        if any(np.isnan(x)) or any(np.isnan(y)) or any(np.isnan(z)):
                                            continue
                                        
                                        # Add an extra point to close the polygon
                                        x.append(x[0])
                                        y.append(y[0])
                                        z.append(z[0])
                                        
                                        # Add face to plot
                                        fig.add_trace(
                                            go.Mesh3d(
                                                x=x, y=y, z=z,
                                                color=color,
                                                opacity=0.7,
                                                showscale=False
                                            )
                                        )
                                    except Exception as e:
                                        print(f"Error adding face: {e}")
                                        continue
                            except Exception as e:
                                print(f"Error processing cell ({i},{j},{k}): {e}")
                                continue
                
                # Update layout for 3D view
                fig.update_layout(
                    title=f"3D Grid Visualization" + (f" - {property_name}" if property_name else ""),
                    scene=dict(
                        xaxis_title="X",
                        yaxis_title="Y",
                        zaxis_title="Z"
                    ),
                    height=800,
                    width=1000
                )
                
                # Add a colorbar if property data is provided
                if property_data is not None:
                    fig.update_layout(
                        coloraxis=dict(
                            colorscale="Viridis",
                            colorbar=dict(
                                title=property_name,
                                thickness=20,
                                len=0.75
                            )
                        )
                    )
            
            # Return the figure if requested
            if return_plotly:
                return fig
            
            # If not returning plotly figure, show it
            fig.show()
            return None
            
        except ImportError:
            # Fall back to matplotlib if plotly is not available
            return self._visualize_grid_matplotlib(
                grid_obj, property_data, property_name, layer, view_mode
            )
            
    def _visualize_grid_matplotlib(
        self,
        grid_obj,
        property_data: Optional[np.ndarray] = None,
        property_name: str = "",
        layer: int = 0,
        view_mode: str = "3D"
    ):
        """Visualize grid using matplotlib when plotly is not available.
        
        Args:
            grid_obj: Eclipse grid object
            property_data: Optional property data to color the grid cells
            property_name: Name of the property data (for label)
            layer: Layer index for 2D view
            view_mode: Visualization mode ('2D' or '3D')
            
        Returns:
            Matplotlib figure
        """
        # Extract grid dimensions
        nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
        
        # Create a helper function to get cell corners
        def get_cell_corners(i, j, k):
            # First check if cell is active before trying to get corners
            try:
                # Check if we have an EclipseReader instance with ACTNUM
                from core.eclipse_reader import EclipseReader
                reader = None
                
                # Try different methods to check if cell is active
                is_active = False
                try:
                    if hasattr(grid_obj, 'active'):
                        is_active = grid_obj.active(i, j, k)
                    elif hasattr(grid_obj, 'is_active'):
                        is_active = grid_obj.is_active(i, j, k)
                    else:
                        # Create a reader and use it to check
                        if reader is None:
                            reader = EclipseReader("")
                        is_active = reader.is_cell_active(grid_obj, i, j, k)
                except Exception:
                    # If we can't determine activity, assume it's active and try anyway
                    is_active = True
                
                # Skip inactive cells
                if not is_active:
                    # Return zeros for inactive cells
                    print(f"Cell ({i},{j},{k}) is inactive, skipping corner calculation")
                    return np.zeros((4, 3))
            
                # Try different methods to get cell corners based on API
                try:
                    # First try direct corner access
                    if hasattr(grid_obj, 'get_cell_corner'):
                        # New API
                        cell_corners = np.zeros((4, 3))
                        for corner_idx in range(4):
                            x, y, z = grid_obj.get_cell_corner(corner_idx, i, j, k)
                            cell_corners[corner_idx] = [x, y, z]
                        return cell_corners
                    elif hasattr(grid_obj, 'getCellCorner'):
                        # Old API
                        cell_corners = np.zeros((4, 3))
                        for corner_idx in range(4):
                            # Pass ijk as a tuple without any parameter name
                            x, y, z = grid_obj.getCellCorner(corner_idx, (i, j, k))
                            cell_corners[corner_idx] = [x, y, z]
                        return cell_corners
                    elif hasattr(grid_obj, 'cell_corners'):
                        # Alternative API - returns all 8 corners, we take first 4
                        all_corners = grid_obj.cell_corners(i, j, k)
                        return all_corners[:4]
                    else:
                        # Try via EclipseReader
                        if reader is None:
                            reader = EclipseReader("")
                        all_corners = reader.get_cell_corners(grid_obj, i, j, k)
                        return all_corners[:4]
                except Exception as e:
                    print(f"Error getting corners for cell ({i},{j},{k}): {e}")
                    # Return zeros as fallback
                    return np.zeros((4, 3))
            except Exception as e:
                print(f"Error in get_cell_corners for cell ({i},{j},{k}): {e}")
                # Return zeros as fallback
                return np.zeros((4, 3))
        
        if view_mode == "2D":
            # Create a 2D figure
            fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            ax = fig.add_subplot(111)
            
            # Create a 2D mesh
            for i in range(nx):
                for j in range(ny):
                    # Skip inactive cells if we can detect them
                    try:
                        if hasattr(grid_obj, 'active') and not grid_obj.active(i, j, layer):
                            continue
                        if hasattr(grid_obj, 'is_active') and not grid_obj.is_active(i, j, layer):
                            continue
                    except:
                        pass
                        
                    # Get the top face corners for this layer
                    cell_corners = get_cell_corners(i, j, layer)
                    
                    # Extract coordinates for plotting
                    x = [cell_corners[k, 0] for k in range(4)] + [cell_corners[0, 0]]
                    y = [cell_corners[k, 1] for k in range(4)] + [cell_corners[0, 1]]
                    
                    # Add color if property data is provided
                    if property_data is not None:
                        try:
                            color = property_data[i, j, layer]
                            if not np.isnan(color):
                                ax.fill(x, y, color=plt.cm.viridis(color), edgecolor='black', linewidth=0.5)
                        except:
                            ax.fill(x, y, facecolor='lightgray', edgecolor='black', linewidth=0.5)
                    else:
                        ax.fill(x, y, facecolor='lightgray', edgecolor='black', linewidth=0.5)
            
            # Set axis labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(f"Grid Layer {layer+1}/{nz}" + (f" - {property_name}" if property_name else ""))
            
            # Add a colorbar if property data is provided
            if property_data is not None:
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis)
                sm.set_array(property_data[:,:,layer])
                plt.colorbar(sm, ax=ax, label=property_name)
            
        else:  # 3D mode
            # Create a 3D figure
            fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
            ax = fig.add_subplot(111, projection='3d')
            
            # Sampling for visualization (plotting all cells can be too dense)
            skip_x = max(1, nx // 20)
            skip_y = max(1, ny // 20)
            skip_z = max(1, nz // 10)
            
            # Create grid points
            x_points, y_points, z_points = [], [], []
            colors = []
            
            # For each grid cell
            for i in range(0, nx, skip_x):
                for j in range(0, ny, skip_y):
                    for k in range(0, nz, skip_z):
                        # Skip inactive cells if we can detect them
                        try:
                            if hasattr(grid_obj, 'active') and not grid_obj.active(i, j, k):
                                continue
                            if hasattr(grid_obj, 'is_active') and not grid_obj.is_active(i, j, k):
                                continue
                        except:
                            pass
                            
                        # Get cell center using corners
                        try:
                            corners = get_cell_corners(i, j, k)
                            center = np.mean(corners, axis=0)
                            
                            # Skip zero or NaN centers
                            if np.all(np.isclose(center, 0)) or np.any(np.isnan(center)):
                                continue
                                
                            x_points.append(center[0])
                            y_points.append(center[1])
                            z_points.append(center[2])
                            
                            # Add color if property data is provided
                            if property_data is not None:
                                try:
                                    color = property_data[i, j, k]
                                    if np.isnan(color):
                                        color = 0.5  # Default gray for NaN values
                                    colors.append(color)
                                except:
                                    colors.append(0.5)
                            else:
                                colors.append(0.5)  # Default gray
                        except Exception as e:
                            print(f"Error processing cell center ({i},{j},{k}): {e}")
                            continue
            
            # Plot the points if we have any
            if x_points:
                if property_data is not None:
                    scatter = ax.scatter(
                        x_points, y_points, z_points,
                        c=colors,
                        cmap='viridis',
                        alpha=0.7,
                        s=10
                    )
                    plt.colorbar(scatter, ax=ax, label=property_name)
                else:
                    ax.scatter(
                        x_points, y_points, z_points,
                        color='lightgray',
                        alpha=0.7,
                        s=10
                    )
            
            # Set axis labels
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"3D Grid Visualization" + (f" - {property_name}" if property_name else ""))
            
        # Add grid and layout settings
        if self.config.show_grid:
            plt.grid(True)
            
        if self.config.tight_layout:
            plt.tight_layout()
            
        return fig
    
    def visualize_property_slice(
        self,
        grid_obj,
        property_data: np.ndarray,
        property_name: str,
        slice_direction: str = "K",
        slice_index: int = 0,
        return_plotly: bool = False
    ):
        """Visualize a slice of a 3D property.
        
        Args:
            grid_obj: Eclipse grid object
            property_data: Property data to visualize (3D array or 2D array if already a slice)
            property_name: Name of the property
            slice_direction: Direction of the slice ('I', 'J', or 'K')
            slice_index: Index of the slice
            return_plotly: Whether to return a plotly figure
            
        Returns:
            Plotly figure or matplotlib figure depending on return_plotly
        """
        try:
            import plotly.graph_objects as go
            import numpy as np
            
            # Check property data dimensionality
            if property_data is None:
                raise ValueError("Property data is required for slice visualization")
            
            # Extract grid dimensions for context
            nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            
            # Handle different dimensions of property data
            if property_data.ndim == 2:
                # Check if this is a flattened property (1, n_active_cells)
                if property_data.shape[0] == 1 and property_data.shape[1] > nx * ny * nz * 0.5:
                    # This is likely a flattened property for active cells only
                    # Create a 3D array filled with NaN
                    property_3d = np.full((nx, ny, nz), np.nan)
                    
                    # Get the active cell indices and values
                    active_indices = []
                    active_values = property_data[0]
                    
                    # Create a reader to help with active cell mapping
                    from core.eclipse_reader import EclipseReader
                    reader = EclipseReader("")
                    
                    # Map active cell values back to 3D grid
                    for i in range(nx):
                        for j in range(ny):
                            for k in range(nz):
                                if reader.is_cell_active(grid_obj, i, j, k):
                                    # Get the active index for this cell
                                    try:
                                        if hasattr(grid_obj, 'get_active_index'):
                                            active_idx = grid_obj.get_active_index(i, j, k)
                                        elif hasattr(grid_obj, 'getActiveIndex'):
                                            active_idx = grid_obj.getActiveIndex(i, j, k)
                                        else:
                                            # If we can't get the active index, try to count active cells
                                            active_idx = len(active_indices)
                                            
                                        if active_idx < len(active_values):
                                            property_3d[i, j, k] = active_values[active_idx]
                                    except Exception as e:
                                        print(f"Warning: Could not map active cell ({i},{j},{k}): {e}")
                    
                    # Use the 3D property data for visualization
                    property_data = property_3d
                else:
                    # Already a 2D slice, use directly
                    slice_data = property_data
                    # Set appropriate labels based on most likely slice orientation
                    dim1, dim2 = property_data.shape
                    
                    # Try to determine the slice direction from the shape
                    if dim1 == nx and dim2 == ny:
                        # Likely an IJ slice (layer)
                        x_label, y_label = "I", "J"
                        title = f"{property_name} - K Layer (2D Property)"
                    elif dim1 == nx and dim2 == nz:
                        # Likely an IK slice
                        x_label, y_label = "I", "K"
                        title = f"{property_name} - J Slice (2D Property)"
                    elif dim1 == ny and dim2 == nz:
                        # Likely a JK slice
                        x_label, y_label = "J", "K"
                        title = f"{property_name} - I Slice (2D Property)"
                    else:
                        # Generic 2D slice
                        x_label, y_label = "Column", "Row"
                        title = f"{property_name} - 2D Property ({dim1}x{dim2})"
            elif property_data.ndim == 3:
                # Extract grid dimensions
                nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
                
                # Check if dimensions match
                if property_data.shape != (nx, ny, nz):
                    print(f"Warning: Property shape {property_data.shape} doesn't match grid dimensions {(nx, ny, nz)}")
                
                # Validate slice index
                if slice_direction == "I" and (slice_index < 0 or slice_index >= nx):
                    raise ValueError(f"Invalid I slice index. Must be between 0 and {nx-1}")
                elif slice_direction == "J" and (slice_index < 0 or slice_index >= ny):
                    raise ValueError(f"Invalid J slice index. Must be between 0 and {ny-1}")
                elif slice_direction == "K" and (slice_index < 0 or slice_index >= nz):
                    raise ValueError(f"Invalid K slice index. Must be between 0 and {nz-1}")
                
                # Extract the slice data based on direction
                if slice_direction == "I":
                    try:
                        slice_data = property_data[slice_index, :, :]
                    except IndexError as e:
                        print(f"Error accessing I slice {slice_index}: {e}")
                        # Create blank slice as fallback
                        slice_data = np.zeros((ny, nz))
                    x_label, y_label = "J", "K"
                    title = f"{property_name} - I={slice_index+1} Slice"
                    
                elif slice_direction == "J":
                    try:
                        slice_data = property_data[:, slice_index, :]
                    except IndexError as e:
                        print(f"Error accessing J slice {slice_index}: {e}")
                        # Create blank slice as fallback
                        slice_data = np.zeros((nx, nz))
                    x_label, y_label = "I", "K"
                    title = f"{property_name} - J={slice_index+1} Slice"
                    
                else:  # K direction
                    try:
                        slice_data = property_data[:, :, slice_index]
                    except IndexError as e:
                        print(f"Error accessing K slice {slice_index}: {e}")
                        # Create blank slice as fallback
                        slice_data = np.zeros((nx, ny))
                    x_label, y_label = "I", "J"
                    title = f"{property_name} - K={slice_index+1} Slice"
            else:
                raise ValueError(f"Property data must be 2D or 3D array, got {property_data.ndim}D")
            
            # Create a figure
            fig = go.Figure()
            
            # Create a heatmap with the slice data
            # Handle NaN values better
            if np.isnan(slice_data).all():
                print(f"Warning: All values in slice are NaN")
                # Create dummy data for visualization
                slice_data = np.zeros(slice_data.shape)
                
            fig.add_trace(
                go.Heatmap(
                    z=slice_data.T,  # Transpose for correct orientation
                    colorscale="Viridis",
                    colorbar=dict(title=property_name),
                    hoverongaps=False,
                    # Set reasonable zmin and zmax excluding NaN values
                    zmin=np.nanmin(slice_data) if not np.isnan(slice_data).all() else 0,
                    zmax=np.nanmax(slice_data) if not np.isnan(slice_data).all() else 1
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=800,
                width=1000
            )
            
            # Return the figure if requested
            if return_plotly:
                return fig
            
            # Try to show the figure with QtWebEngineWidgets
            try:
                # Show the figure
                from PyQt6.QtWebEngineWidgets import QWebEngineView
                from plotly.offline import plot
                import os
                
                # Make sure sandbox is disabled for network paths
                os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
                
                # Convert to HTML
                html = plot(fig, include_plotlyjs=True, output_type='div')
                
                # Create a simple HTML page with the div
                full_html = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <meta charset="utf-8">
                    <title>{title}</title>
                </head>
                <body>
                    {html}
                </body>
                </html>
                """
                
                # Create QApplication if needed
                from PyQt6.QtWidgets import QApplication
                if not QApplication.instance():
                    app = QApplication([])
                
                # Create web view
                web_view = QWebEngineView()
                web_view.setHtml(full_html)
                web_view.resize(1000, 800)
                web_view.setWindowTitle(title)
                web_view.show()
                
                # Keep reference to prevent garbage collection
                self._web_view_ref = web_view
                
                return None
            except Exception as e:
                print(f"Warning: Could not show interactive plot: {e}. Falling back to matplotlib.")
                # Fall back to matplotlib
                return self._visualize_property_slice_matplotlib(
                    grid_obj, property_data, property_name, slice_direction, slice_index
                )
            
        except ImportError as e:
            # Fall back to matplotlib
            print(f"Warning: Could not use Plotly: {e}. Falling back to matplotlib.")
            return self._visualize_property_slice_matplotlib(
                grid_obj, property_data, property_name, slice_direction, slice_index
            )
    
    def _visualize_property_slice_matplotlib(
        self,
        grid_obj,
        property_data: np.ndarray,
        property_name: str,
        slice_direction: str = "K",
        slice_index: int = 0
    ):
        """Visualize a slice of a 3D property using matplotlib.
        
        Args:
            grid_obj: Eclipse grid object
            property_data: Property data to visualize
            property_name: Name of the property
            slice_direction: Direction of the slice ('I', 'J', or 'K')
            slice_index: Index of the slice
            
        Returns:
            Matplotlib figure
        """
        # Check property data dimensionality
        if property_data is None:
            raise ValueError("Property data is required for slice visualization")
            
        # Create figure
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Extract grid dimensions for context
        nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
        
        # Handle different dimensions of property data
        if property_data.ndim == 2:
            # Check if this might be a time series property (timesteps, cells)
            if property_data.shape[1] == grid_obj.getNumActive() or property_data.shape[1] > nx * ny * nz * 0.5:
                # This is likely a time series property with (timesteps, active_cells)
                # Create a special view showing the last timestep values
                last_timestep = property_data[-1, :]
                
                # Create a plot with active cell indices on x-axis
                ax = fig.add_subplot(111)
                
                # Plot a scatter with values colored by property value
                sc = ax.scatter(
                    np.arange(len(last_timestep)),
                    np.ones(len(last_timestep)),
                    c=last_timestep,
                    cmap='viridis',
                    alpha=0.7,
                    s=10
                )
                
                # Add colorbar
                plt.colorbar(sc, ax=ax, label=property_name)
                
                # Set labels and title
                ax.set_xlabel("Cell Index")
                ax.set_yticks([])  # Hide y-axis ticks
                ax.set_title(f"{property_name} - Last Timestep Values for {len(last_timestep)} Active Cells")
                
                # Add grid settings
                if self.config.show_grid:
                    ax.grid(True, axis='x')
                    
                if self.config.tight_layout:
                    plt.tight_layout()
                    
                return fig
            
            # Already a 2D slice, use directly
            slice_data = property_data
            ax = fig.add_subplot(111)
            
            # Set appropriate labels based on most likely slice orientation
            dim1, dim2 = property_data.shape
            
            # Try to determine the slice direction from the shape
            if dim1 == nx and dim2 == ny:
                # Likely an IJ slice (layer)
                x_label, y_label = "I", "J"
                title = f"{property_name} - K Layer (2D Property)"
            elif dim1 == nx and dim2 == nz:
                # Likely an IK slice
                x_label, y_label = "I", "K"
                title = f"{property_name} - J Slice (2D Property)"
            elif dim1 == ny and dim2 == nz:
                # Likely a JK slice
                x_label, y_label = "J", "K"
                title = f"{property_name} - I Slice (2D Property)"
            else:
                # Generic 2D slice
                x_label, y_label = "Column", "Row"
                title = f"{property_name} - 2D Property ({dim1}x{dim2})"
        elif property_data.ndim == 3:
            # Extract grid dimensions
            nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            
            # Extract the slice data based on direction
            if slice_direction == "I":
                try:
                    slice_data = property_data[slice_index, :, :]
                except IndexError:
                    # Create blank slice as fallback
                    slice_data = np.zeros((ny, nz))
                x_label, y_label = "J", "K"
                title = f"{property_name} - I={slice_index+1} Slice"
                
            elif slice_direction == "J":
                try:
                    slice_data = property_data[:, slice_index, :]
                except IndexError:
                    # Create blank slice as fallback
                    slice_data = np.zeros((nx, nz))
                x_label, y_label = "I", "K"
                title = f"{property_name} - J={slice_index+1} Slice"
                
            else:  # K direction
                try:
                    slice_data = property_data[:, :, slice_index]
                except IndexError:
                    # Create blank slice as fallback
                    slice_data = np.zeros((nx, ny))
                x_label, y_label = "I", "J"
                title = f"{property_name} - K={slice_index+1} Slice"
        else:
            raise ValueError(f"Property data must be 2D or 3D array, got {property_data.ndim}D")
        
        # Create a heatmap - handle NaN values
        if np.isnan(slice_data).all():
            # All NaN, create dummy data
            slice_data = np.zeros(slice_data.shape)
        
        # Create a heatmap
        im = ax.imshow(
            slice_data.T,  # Transpose for correct orientation
            cmap='viridis',
            origin='lower',
            aspect='auto'
        )
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=property_name)
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Add grid settings
        if self.config.show_grid:
            ax.grid(False)  # Grid not useful with imshow
            
        if self.config.tight_layout:
            plt.tight_layout()
            
        return fig

    def visualize_active_cells(
        self,
        grid_obj,
        actnum_data: Optional[np.ndarray] = None,
        slice_direction: str = "K",
        slice_index: int = 0,
        return_plotly: bool = False
    ):
        """Visualize active cells in the grid.
        
        Args:
            grid_obj: Eclipse grid object
            actnum_data: Optional ACTNUM data (1 for active, 0 for inactive)
            slice_direction: Direction of the slice ('I', 'J', or 'K')
            slice_index: Index of the slice
            return_plotly: Whether to return a plotly figure
            
        Returns:
            Plotly figure or None if not return_plotly
        """
        try:
            import plotly.graph_objects as go
            
            # Extract grid dimensions
            nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
            
            # If ACTNUM data is not provided, try to generate it
            if actnum_data is None:
                # Create a reader to help with this
                from core.eclipse_reader import EclipseReader
                reader = EclipseReader("")
                
                # Try to read ACTNUM from file
                try:
                    # Look for ACTNUM.grdecl in the same directory as the grid file
                    if hasattr(grid_obj, 'getName'):
                        grid_path = grid_obj.getName()
                        grid_dir = Path(grid_path).parent
                        actnum_file = grid_dir / "ACTNUM.grdecl"
                        actnum_data = reader.read_actnum_file(actnum_file)
                    
                    # If not found, generate from grid object
                    if actnum_data is None:
                        # Create an array of ones (all active)
                        actnum_data = np.ones((nx, ny, nz), dtype=np.int32)
                        
                        # Mark inactive cells
                        for i in range(nx):
                            for j in range(ny):
                                for k in range(nz):
                                    try:
                                        is_active = reader.is_cell_active(grid_obj, i, j, k)
                                        if not is_active:
                                            actnum_data[i, j, k] = 0
                                    except:
                                        # If we can't determine, assume inactive
                                        actnum_data[i, j, k] = 0
                except Exception as e:
                    print(f"Error generating ACTNUM data: {e}")
                    # Create default all active
                    actnum_data = np.ones((nx, ny, nz), dtype=np.int32)
            
            # Ensure ACTNUM data has correct shape
            if actnum_data.ndim == 1:
                # Reshape 1D ACTNUM to 3D
                try:
                    actnum_3d = np.ones((nx, ny, nz), dtype=np.int32)
                    idx = 0
                    for k in range(nz):
                        for j in range(ny):
                            for i in range(nx):
                                if idx < len(actnum_data):
                                    actnum_3d[i, j, k] = actnum_data[idx]
                                    idx += 1
                    actnum_data = actnum_3d
                except Exception as e:
                    print(f"Error reshaping ACTNUM data: {e}")
            
            # Extract slice based on direction
            if slice_direction == "I":
                if slice_index < nx:
                    slice_data = actnum_data[slice_index, :, :]
                    title = f"Active Cells - I={slice_index+1} Slice"
                    x_label, y_label = "J", "K"
                else:
                    print(f"Invalid I slice index {slice_index}, using 0")
                    slice_data = actnum_data[0, :, :]
                    title = f"Active Cells - I=1 Slice"
                    x_label, y_label = "J", "K"
            elif slice_direction == "J":
                if slice_index < ny:
                    slice_data = actnum_data[:, slice_index, :]
                    title = f"Active Cells - J={slice_index+1} Slice"
                    x_label, y_label = "I", "K"
                else:
                    print(f"Invalid J slice index {slice_index}, using 0")
                    slice_data = actnum_data[:, 0, :]
                    title = f"Active Cells - J=1 Slice"
                    x_label, y_label = "I", "K"
            else:  # K direction
                if slice_index < nz:
                    slice_data = actnum_data[:, :, slice_index]
                    title = f"Active Cells - K={slice_index+1} Slice"
                    x_label, y_label = "I", "J"
                else:
                    print(f"Invalid K slice index {slice_index}, using 0")
                    slice_data = actnum_data[:, :, 0]
                    title = f"Active Cells - K=1 Slice"
                    x_label, y_label = "I", "J"
            
            # Create a figure
            fig = go.Figure()
            
            # Create a heatmap with the slice data
            fig.add_trace(
                go.Heatmap(
                    z=slice_data.T,  # Transpose for correct orientation
                    colorscale=[
                        [0, 'rgb(255,0,0)'],  # Red for inactive (0)
                        [1, 'rgb(0,255,0)']   # Green for active (1)
                    ],
                    colorbar=dict(
                        title="Active Status",
                        tickvals=[0, 1],
                        ticktext=["Inactive", "Active"]
                    ),
                    hoverongaps=False,
                    zmin=0,
                    zmax=1,
                    hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>Active: %{{z}}<extra></extra>'
                )
            )
            
            # Update layout
            fig.update_layout(
                title=title,
                xaxis_title=x_label,
                yaxis_title=y_label,
                height=800,
                width=1000
            )
            
            # Return the figure if requested
            if return_plotly:
                return fig
            
            # Show the figure
            fig.show()
            return None
            
        except ImportError:
            # Fall back to matplotlib if plotly is not available
            return self._visualize_active_cells_matplotlib(
                grid_obj, actnum_data, slice_direction, slice_index
            )
        
    def _visualize_active_cells_matplotlib(
        self,
        grid_obj,
        actnum_data: Optional[np.ndarray] = None,
        slice_direction: str = "K",
        slice_index: int = 0
    ):
        """Visualize active cells using matplotlib.
        
        Args:
            grid_obj: Eclipse grid object
            actnum_data: Optional ACTNUM data (1 for active, 0 for inactive)
            slice_direction: Direction of the slice ('I', 'J', or 'K')
            slice_index: Index of the slice
            
        Returns:
            Matplotlib figure
        """
        # Extract grid dimensions
        nx, ny, nz = grid_obj.getNX(), grid_obj.getNY(), grid_obj.getNZ()
        
        # If ACTNUM data is not provided, try to generate it
        if actnum_data is None:
            # Create a reader to help with this
            from core.eclipse_reader import EclipseReader
            reader = EclipseReader("")
            
            # Try to read ACTNUM from file
            try:
                # Look for ACTNUM.grdecl in the same directory as the grid file
                if hasattr(grid_obj, 'getName'):
                    grid_path = grid_obj.getName()
                    grid_dir = Path(grid_path).parent
                    actnum_file = grid_dir / "ACTNUM.grdecl"
                    actnum_data = reader.read_actnum_file(actnum_file)
                
                # If not found, generate from grid object
                if actnum_data is None:
                    # Create an array of ones (all active)
                    actnum_data = np.ones((nx, ny, nz), dtype=np.int32)
                    
                    # Mark inactive cells
                    for i in range(nx):
                        for j in range(ny):
                            for k in range(nz):
                                try:
                                    is_active = reader.is_cell_active(grid_obj, i, j, k)
                                    if not is_active:
                                        actnum_data[i, j, k] = 0
                                except:
                                    # If we can't determine, assume inactive
                                    actnum_data[i, j, k] = 0
            except Exception as e:
                print(f"Error generating ACTNUM data: {e}")
                # Create default all active
                actnum_data = np.ones((nx, ny, nz), dtype=np.int32)
        
        # Ensure ACTNUM data has correct shape
        if actnum_data.ndim == 1:
            # Reshape 1D ACTNUM to 3D
            try:
                actnum_3d = np.ones((nx, ny, nz), dtype=np.int32)
                idx = 0
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx):
                            if idx < len(actnum_data):
                                actnum_3d[i, j, k] = actnum_data[idx]
                                idx += 1
                actnum_data = actnum_3d
            except Exception as e:
                print(f"Error reshaping ACTNUM data: {e}")
        
        # Extract slice based on direction
        if slice_direction == "I":
            if slice_index < nx:
                slice_data = actnum_data[slice_index, :, :]
                title = f"Active Cells - I={slice_index+1} Slice"
                x_label, y_label = "J", "K"
            else:
                print(f"Invalid I slice index {slice_index}, using 0")
                slice_data = actnum_data[0, :, :]
                title = f"Active Cells - I=1 Slice"
                x_label, y_label = "J", "K"
        elif slice_direction == "J":
            if slice_index < ny:
                slice_data = actnum_data[:, slice_index, :]
                title = f"Active Cells - J={slice_index+1} Slice"
                x_label, y_label = "I", "K"
            else:
                print(f"Invalid J slice index {slice_index}, using 0")
                slice_data = actnum_data[:, 0, :]
                title = f"Active Cells - J=1 Slice"
                x_label, y_label = "I", "K"
        else:  # K direction
            if slice_index < nz:
                slice_data = actnum_data[:, :, slice_index]
                title = f"Active Cells - K={slice_index+1} Slice"
                x_label, y_label = "I", "J"
            else:
                print(f"Invalid K slice index {slice_index}, using 0")
                slice_data = actnum_data[:, :, 0]
                title = f"Active Cells - K=1 Slice"
                x_label, y_label = "I", "J"
        
        # Create figure
        fig = plt.figure(figsize=self.config.figure_size, dpi=self.config.dpi)
        ax = fig.add_subplot(111)
        
        # Create a mask for active cells (1) and inactive cells (0)
        cmap = plt.cm.colors.ListedColormap(['red', 'green'])
        
        # Plot the heatmap
        im = ax.imshow(
            slice_data.T,  # Transpose for correct orientation
            cmap=cmap,
            origin='lower',
            vmin=0,
            vmax=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_ticks([0.25, 0.75])
        cbar.set_ticklabels(['Inactive', 'Active'])
        
        # Set labels and title
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        
        # Add grid settings
        if self.config.tight_layout:
            plt.tight_layout()
            
        return fig

class ReservoirAnalyzer:
    """Analysis tools for reservoir simulation results."""
    
    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass
        
    def compute_metrics(
        self,
        true_field: Union[np.ndarray, torch.Tensor],
        pred_field: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute error metrics.
        
        Args:
            true_field: True field data
            pred_field: Predicted field data
            
        Returns:
            Dictionary of error metrics
        """
        # Convert to numpy if needed
        if isinstance(true_field, torch.Tensor):
            true_field = true_field.detach().cpu().numpy()
        if isinstance(pred_field, torch.Tensor):
            pred_field = pred_field.detach().cpu().numpy()
            
        # Compute error
        error = true_field - pred_field
        
        # Compute metrics
        metrics = {
            "mse": np.mean(error ** 2),
            "rmse": np.sqrt(np.mean(error ** 2)),
            "mae": np.mean(np.abs(error)),
            "max_error": np.max(np.abs(error)),
            "r2": 1 - np.sum(error ** 2) / np.sum((true_field - np.mean(true_field)) ** 2)
        }
        
        return metrics
        
    def compute_statistics(
        self,
        field: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, float]:
        """Compute field statistics.
        
        Args:
            field: Field data
            
        Returns:
            Dictionary of statistics
        """
        # Convert to numpy if needed
        if isinstance(field, torch.Tensor):
            field = field.detach().cpu().numpy()
            
        # Compute statistics
        stats = {
            "mean": np.mean(field),
            "std": np.std(field),
            "min": np.min(field),
            "max": np.max(field),
            "median": np.median(field),
            "q1": np.percentile(field, 25),
            "q3": np.percentile(field, 75)
        }
        
        return stats 