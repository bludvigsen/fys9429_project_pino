"""Physics constraints for the PINO model.

This module provides physics-based constraints and loss functions
for training the PINO model on reservoir simulation data.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    """Configuration for physics constraints."""
    
    # Conservation laws
    enforce_mass_conservation: bool = True
    enforce_energy_conservation: bool = True
    enforce_momentum_conservation: bool = True
    
    # Physical constraints
    enforce_positive_pressure: bool = True
    enforce_saturation_bounds: bool = True
    enforce_porosity_bounds: bool = True
    enforce_permeability_bounds: bool = True
    
    # Loss weights
    mass_weight: float = 1.0
    energy_weight: float = 1.0
    momentum_weight: float = 1.0
    pressure_weight: float = 1.0
    saturation_weight: float = 1.0
    porosity_weight: float = 1.0
    permeability_weight: float = 1.0
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if any(w < 0 for w in [
            self.mass_weight,
            self.energy_weight,
            self.momentum_weight,
            self.pressure_weight,
            self.saturation_weight,
            self.porosity_weight,
            self.permeability_weight
        ]):
            raise ValueError("All weights must be non-negative")

class PhysicsLoss(nn.Module):
    """Physics-based loss functions for PINO training."""
    
    def __init__(self, config: PhysicsConfig) -> None:
        """Initialize the physics loss module.
        
        Args:
            config: Physics configuration
        """
        super().__init__()
        self.config = config
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        inputs: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute physics-based loss.
        
        Args:
            outputs: Dictionary of model outputs
            inputs: Optional dictionary of model inputs
            
        Returns:
            Total physics loss
        """
        losses = []
        
        # Conservation laws
        if self.config.enforce_mass_conservation:
            mass_loss = self._mass_conservation_loss(outputs)
            losses.append(self.config.mass_weight * mass_loss)
            
        if self.config.enforce_energy_conservation:
            energy_loss = self._energy_conservation_loss(outputs)
            losses.append(self.config.energy_weight * energy_loss)
            
        if self.config.enforce_momentum_conservation:
            momentum_loss = self._momentum_conservation_loss(outputs)
            losses.append(self.config.momentum_weight * momentum_loss)
            
        # Physical constraints
        if self.config.enforce_positive_pressure:
            pressure_loss = self._positive_pressure_loss(outputs)
            losses.append(self.config.pressure_weight * pressure_loss)
            
        if self.config.enforce_saturation_bounds:
            saturation_loss = self._saturation_bounds_loss(outputs)
            losses.append(self.config.saturation_weight * saturation_loss)
            
        if self.config.enforce_porosity_bounds:
            porosity_loss = self._porosity_bounds_loss(outputs)
            losses.append(self.config.porosity_weight * porosity_loss)
            
        if self.config.enforce_permeability_bounds:
            permeability_loss = self._permeability_bounds_loss(outputs)
            losses.append(self.config.permeability_weight * permeability_loss)
            
        return torch.stack(losses).sum()
        
    def _mass_conservation_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute mass conservation loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Mass conservation loss
        """
        # Extract relevant variables
        pressure = outputs.get("Pressure")
        water_sat = outputs.get("Water_saturation")
        oil_sat = outputs.get("Oil_saturation")
        gas_sat = outputs.get("Gas_saturation")
        
        if all(v is not None for v in [pressure, water_sat, oil_sat, gas_sat]):
            # Compute mass conservation residual
            # This is a simplified version - actual implementation would depend on
            # the specific physics of the reservoir
            mass_residual = (
                water_sat + oil_sat + gas_sat - 1.0
            ).abs().mean()
            
            return mass_residual
        else:
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
            
    def _energy_conservation_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute energy conservation loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Energy conservation loss
        """
        # Extract relevant variables
        pressure = outputs.get("Pressure")
        temperature = outputs.get("Temperature")
        
        if all(v is not None for v in [pressure, temperature]):
            # Compute energy conservation residual
            # This is a simplified version - actual implementation would depend on
            # the specific physics of the reservoir
            energy_residual = (
                pressure * temperature
            ).abs().mean()
            
            return energy_residual
        else:
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
            
    def _momentum_conservation_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute momentum conservation loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Momentum conservation loss
        """
        # Extract relevant variables
        pressure = outputs.get("Pressure")
        velocity = outputs.get("Velocity")
        
        if all(v is not None for v in [pressure, velocity]):
            # Compute momentum conservation residual
            # This is a simplified version - actual implementation would depend on
            # the specific physics of the reservoir
            momentum_residual = (
                pressure * velocity
            ).abs().mean()
            
            return momentum_residual
        else:
            return torch.tensor(0.0, device=next(iter(outputs.values())).device)
            
    def _positive_pressure_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute positive pressure constraint loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Positive pressure constraint loss
        """
        pressure = outputs.get("Pressure")
        if pressure is not None:
            return torch.relu(-pressure).mean()
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
    def _saturation_bounds_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute saturation bounds constraint loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Saturation bounds constraint loss
        """
        losses = []
        
        for sat_name in ["Water_saturation", "Oil_saturation", "Gas_saturation"]:
            saturation = outputs.get(sat_name)
            if saturation is not None:
                # Saturations should be between 0 and 1
                losses.append(
                    torch.relu(-saturation).mean() +
                    torch.relu(saturation - 1.0).mean()
                )
                
        return torch.stack(losses).sum() if losses else torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
    def _porosity_bounds_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute porosity bounds constraint loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Porosity bounds constraint loss
        """
        porosity = outputs.get("Porosity")
        if porosity is not None:
            # Porosity should be between 0 and 1
            return (
                torch.relu(-porosity).mean() +
                torch.relu(porosity - 1.0).mean()
            )
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
    def _permeability_bounds_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute permeability bounds constraint loss.
        
        Args:
            outputs: Dictionary of model outputs
            
        Returns:
            Permeability bounds constraint loss
        """
        permeability = outputs.get("Permeability")
        if permeability is not None:
            # Permeability should be positive
            return torch.relu(-permeability).mean()
        return torch.tensor(0.0, device=next(iter(outputs.values())).device)

class DarcyLaw(nn.Module):
    """Darcy's law implementation for fluid flow."""
    
    def __init__(self) -> None:
        """Initialize Darcy's law module."""
        super().__init__()
        
    def forward(
        self,
        permeability: torch.Tensor,
        pressure_gradient: torch.Tensor,
        viscosity: torch.Tensor,
        density: torch.Tensor
    ) -> torch.Tensor:
        """Compute fluid velocity using Darcy's law.
        
        Args:
            permeability: Permeability tensor
            pressure_gradient: Pressure gradient
            viscosity: Fluid viscosity
            density: Fluid density
            
        Returns:
            Fluid velocity
        """
        # Darcy's law: v = -k/μ * (∇p + ρg)
        # where:
        # v: velocity
        # k: permeability
        # μ: viscosity
        # p: pressure
        # ρ: density
        # g: gravitational acceleration (assumed to be in z-direction)
        
        # Compute velocity
        velocity = -permeability / viscosity * (pressure_gradient + density * 9.81)
        
        return velocity 