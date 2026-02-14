"""
Tutorial 4: DDPM on 2D Toy Data

This package implements a complete denoising diffusion probabilistic model
for 2D toy datasets, designed for educational purposes.
"""

__version__ = "0.1.0"

from .models import SimpleMLPDenoiser
from .diffusion import GaussianDiffusion
from .utils import create_toy_dataset
from .visualization import (
    visualize_forward_process, 
    visualize_samples,
    visualize_marginal_distributions,
)

__all__ = [
    "SimpleMLPDenoiser",
    "GaussianDiffusion",
    "create_toy_dataset",
    "visualize_forward_process",
    "visualize_samples",
    "visualize_marginal_distributions",
]
