"""
Gaussian diffusion process for DDPM.

This module implements the forward noising process and reverse denoising process
following the DDPM paper: "Denoising Diffusion Probabilistic Models"
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class GaussianDiffusion:
    """
    Gaussian Diffusion Process.
    
    Forward process (adding noise):
        q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1 - ᾱ_t) I)
    
    Reverse process (denoising):
        p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), σ_t² I)
    
    where:
        - β_t: variance schedule
        - α_t = 1 - β_t
        - ᾱ_t = ∏_{s=1}^t α_s
    """
    
    def __init__(
        self,
        n_steps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "linear",
        device: str = "cpu",
    ):
        """
        Args:
            n_steps: Number of diffusion timesteps T
            beta_start: Starting value of β_1
            beta_end: Ending value of β_T
            schedule: Noise schedule type ("linear" or "cosine")
            device: Device to run on
        """
        self.n_steps = n_steps
        self.device = device
        
        # Create noise schedule
        if schedule == "linear":
            self.betas = self._linear_beta_schedule(beta_start, beta_end, n_steps)
        elif schedule == "cosine":
            self.betas = self._cosine_beta_schedule(n_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = torch.cat([torch.tensor([1.0]), self.alpha_bars[:-1]])
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        
        # Move to device
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.alpha_bars_prev = self.alpha_bars_prev.to(device)
        self.posterior_variance = self.posterior_variance.to(device)
    
    def _linear_beta_schedule(
        self, beta_start: float, beta_end: float, n_steps: int
    ) -> torch.Tensor:
        """
        Linear noise schedule.
        
        β_t increases linearly from β_start to β_end.
        """
        return torch.linspace(beta_start, beta_end, n_steps)
    
    def _cosine_beta_schedule(self, n_steps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule (from "Improved DDPM" paper).
        
        This often works better than linear schedule.
        """
        steps = n_steps + 1
        x = torch.linspace(0, n_steps, steps)
        alphas_cumprod = torch.cos(((x / n_steps) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: Sample from q(x_t | x_0).
        
        x_t = √(ᾱ_t) x_0 + √(1 - ᾱ_t) ε
        
        where ε ~ N(0, I)
        
        Args:
            x_0: Clean data, shape (batch_size, dim)
            t: Timesteps, shape (batch_size,)
            noise: Optional pre-sampled noise
        
        Returns:
            x_t: Noisy samples
            noise: The noise that was added
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        
        # Get √(ᾱ_t) and √(1 - ᾱ_t)
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t])[:, None]
        sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alpha_bars[t])[:, None]
        
        # Apply formula
        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        
        return x_t, noise
    
    def p_sample(
        self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, t_index: int
    ) -> torch.Tensor:
        """
        Reverse diffusion: Sample from p_θ(x_{t-1} | x_t).
        
        This is the denoising step.
        
        Args:
            model: Trained denoiser network
            x_t: Noisy samples at timestep t
            t: Current timesteps (tensor of same value)
            t_index: Index of timestep (0 to n_steps-1)
        
        Returns:
            x_{t-1}: Slightly less noisy samples
        """
        # Predict noise
        with torch.no_grad():
            noise_pred = model(x_t, t)
        
        # Get coefficients
        alpha_t = self.alphas[t_index]
        alpha_bar_t = self.alpha_bars[t_index]
        beta_t = self.betas[t_index]
        
        # Compute mean: μ_θ(x_t, t)
        # Formula: (1 / √α_t) * (x_t - (β_t / √(1 - ᾱ_t)) * ε_θ(x_t, t))
        mean = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * noise_pred
        )
        
        # Add noise (except for t=0)
        if t_index > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t_index]
            x_t_minus_1 = mean + torch.sqrt(variance) * noise
        else:
            x_t_minus_1 = mean
        
        return x_t_minus_1
    
    @torch.no_grad()
    def sample(
        self, model: nn.Module, shape: Tuple[int, ...], return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Start from pure noise x_T ~ N(0, I) and iteratively denoise
        to get x_0 ~ p_θ(x_0).
        
        Args:
            model: Trained denoiser
            shape: Shape of samples to generate (batch_size, dim)
            return_trajectory: If True, return all intermediate steps
        
        Returns:
            Generated samples, or trajectory if return_trajectory=True
        """
        model.eval()
        
        batch_size = shape[0]
        
        # Start from pure noise: x_T ~ N(0, I)
        x = torch.randn(shape, device=self.device)
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        # Iteratively denoise: T → T-1 → ... → 1 → 0
        for t_index in reversed(range(self.n_steps)):
            # Create timestep tensor (same timestep for all samples in batch)
            t = torch.full((batch_size,), t_index, device=self.device, dtype=torch.long)
            
            # Denoise one step
            x = self.p_sample(model, x, t, t_index)
            
            if return_trajectory:
                trajectory.append(x.cpu().numpy())
        
        if return_trajectory:
            return np.array(trajectory)
        else:
            return x
    
    def training_loss(
        self, model: nn.Module, x_0: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the denoising loss for training.
        
        Loss = E_t,x_0,ε [ ||ε - ε_θ(x_t, t)||² ]
        
        This is the simple objective from the DDPM paper.
        
        Args:
            model: Denoiser network
            x_0: Clean data samples
        
        Returns:
            Loss scalar
        """
        batch_size = x_0.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.n_steps, (batch_size,), device=self.device)
        
        # Sample noise
        noise = torch.randn_like(x_0)
        
        # Add noise: get x_t
        x_t, _ = self.q_sample(x_0, t, noise)
        
        # Predict noise
        noise_pred = model(x_t, t)
        
        # Compute MSE loss
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        
        return loss


def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Helper function to extract values from a 1-D array according to timesteps,
    then broadcast to match the shape of a tensor.
    """
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)
