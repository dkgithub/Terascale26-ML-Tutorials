"""
Tutorial 2: Flow Matching - From SDEs to ODEs

This package implements conditional flow matching for generative modeling,
demonstrating the ODE formulation as a comparison to DDPM's SDE approach.

Key Learning Points:
- Probability paths and their design
- Velocity fields for ODE-based generation
- Comparison with stochastic diffusion (DDPM)
- ODE solver selection and trade-offs
"""

__version__ = "0.1.0"

from .models import SimpleMLPDenoiser
from .flow import ConditionalFlowMatching, check_implementation
from .utils import create_toy_dataset, ToyDataLoader, set_seed, get_device, count_parameters
from .visualization import (
    visualize_probability_paths,
    visualize_velocity_field,
    visualize_samples,
    plot_training_curves,
    visualize_reverse_process_trajectory,
    create_reverse_process_animation,
    visualize_marginal_distributions,
    compare_ode_solvers,
)

__all__ = [
    "SimpleMLPDenoiser",
    "ConditionalFlowMatching",
    "check_implementation",
    "create_toy_dataset",
    "ToyDataLoader",
    "set_seed",
    "get_device",
    "count_parameters",
    "visualize_probability_paths",
    "visualize_velocity_field",
    "visualize_samples",
    "plot_training_curves",
    "visualize_reverse_process_trajectory",
    "create_reverse_process_animation",
    "visualize_marginal_distributions",
    "compare_ode_solvers",
]
