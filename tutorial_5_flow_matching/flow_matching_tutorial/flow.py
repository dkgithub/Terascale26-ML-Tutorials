"""
Conditional Flow Matching for generative modeling.

This module implements flow matching with probability paths and velocity fields.
Key difference from DDPM: ODE-based (deterministic) instead of SDE-based (stochastic).

LEARNING NOTE: This file contains TODOs for you to complete!
Read the lecture materials and implement the missing pieces.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Literal
from torchdiffeq import odeint


class ConditionalFlowMatching:
    """
    Conditional Flow Matching: ODE-based generative modeling.
    
    Key Concepts:
    1. Probability Path: p_t(x) interpolates from p_0 (noise) to p_1 (data)
    2. Velocity Field: v_t(x) defines how points flow along the path
    3. ODE: dx/dt = v_t(x) generates samples deterministically
    
    Compare with DDPM (Tutorial 1):
    - DDPM: SDE with dx = f(x,t)dt + g(t)dW
    - Flow: ODE with dx/dt = v_t(x)
    
    Args:
        path_type: Type of probability path ("linear" or "optimal_transport")
        sigma_min: Minimum noise level for numerical stability
        device: Compute device
    """
    
    def __init__(
        self,
        path_type: Literal["linear", "optimal_transport"] = "optimal_transport",
        sigma_min: float = 1e-4,
        device: str = "cpu",
    ):
        self.path_type = path_type
        self.sigma_min = sigma_min
        self.device = device
    
    def sample_probability_path(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample from conditional probability path p_t(x | x_0, x_1).
        
        This defines how we interpolate between noise (x_0) and data (x_1).
        
        ============================================================
        TODO #1: IMPLEMENT PROBABILITY PATHS
        ============================================================
        
        Your task: Implement two probability path types.
        
        1. LINEAR PATH:
           Formula: x_t = (1-t) * x_0 + t * x_1 + sigma_t * epsilon
           - Simple linear interpolation
           - Works but not optimal
        
        2. OPTIMAL TRANSPORT PATH:
           Formula: x_t = t * x_1 + (1-t) * x_0
           - Straighter paths in probability space
           - Often better sample quality
        
        Hints:
        - t is shape (batch_size,), expand for broadcasting
        - sigma_t can be small constant (self.sigma_min)
        - epsilon ~ N(0,I) is optional noise for numerical stability
        
        Compare these paths:
        - Which gives straighter trajectories?
        - Which is easier to learn?
        - Try both and see the difference!
        
        ============================================================
        """
        
        # Get batch size and dimensions
        batch_size = x_0.shape[0]
        
        # Expand t for broadcasting: (batch_size,) -> (batch_size, 1)
        t = t.view(batch_size, 1)
        
        # TODO: Implement probability path based on self.path_type
        if self.path_type == "linear":
            # TODO: Implement linear interpolation
            # Hint: x_t = (1-t) * x_0 + t * x_1 + sigma * noise
            raise NotImplementedError(
                "TODO: Implement linear probability path!\n"
                "Formula: x_t = (1-t) * x_0 + t * x_1 + sigma_t * epsilon\n"
                "This is a warm-up - implement the simple version first."
            )
            
        elif self.path_type == "optimal_transport":
            # TODO: Implement optimal transport path
            # Hint: Very similar to linear, but cleaner (no noise needed)
            # This is the recommended path for flow matching
            raise NotImplementedError(
                "TODO: Implement optimal transport probability path!\n"
                "Formula: x_t = t * x_1 + (1-t) * x_0\n"
                "Notice: Straighter paths, no noise term needed!"
            )
        
        else:
            raise ValueError(f"Unknown path type: {self.path_type}")
    
    def compute_target_velocity(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the target velocity field u_t(x_t | x_0, x_1).
        
        This is what the neural network will learn to predict!
        
        ============================================================
        TODO #2: DERIVE TARGET VELOCITY
        ============================================================
        
        Your task: Compute the analytical velocity field.
        
        Key insight: The velocity is the time derivative of the path!
        
        For probability path x_t, the velocity is:
           u_t = dx_t/dt
        
        For LINEAR path: x_t = (1-t)*x_0 + t*x_1
           Taking derivative: u_t = d/dt[(1-t)*x_0 + t*x_1]
                                  = -x_0 + x_1
                                  = x_1 - x_0
        
        For OPTIMAL TRANSPORT: x_t = t*x_1 + (1-t)*x_0  
           Taking derivative: u_t = d/dt[t*x_1 + (1-t)*x_0]
                                  = x_1 - x_0
        
        Interesting: Both give the same velocity!
        
        Question to ponder: If velocities are the same, why do paths matter?
        Answer: The path determines WHERE the velocity is evaluated!
        
        ============================================================
        """
        
        # TODO: Implement target velocity computation
        # Hint: It's simpler than you think - just the derivative!
        raise NotImplementedError(
            "TODO: Compute target velocity field!\n"
            "Hint: For most paths, u_t = x_1 - x_0 (the direction to go)\n"
            "Think: What's the derivative of (1-t)*x_0 + t*x_1 with respect to t?"
        )
    
    def training_loss(
        self,
        model: nn.Module,
        x_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute conditional flow matching loss.
        
        ============================================================
        TODO #3: IMPLEMENT FLOW MATCHING LOSS
        ============================================================
        
        Your task: Implement the training objective.
        
        The loss is:
           L = E_{t, x_0, x_1} [ ||v_θ(x_t, t) - u_t(x_t | x_0, x_1)||² ]
        
        Where:
        - v_θ(x_t, t) is the predicted velocity from the model
        - u_t is the target velocity (computed above)
        - x_t is sampled from the probability path
        
        Algorithm:
        1. Sample x_0 ~ N(0, I) (Gaussian noise)
        2. Sample t ~ Uniform(0, 1)
        3. Compute x_t from probability path
        4. Compute target velocity u_t
        5. Predict velocity v_θ(x_t, t)
        6. Compute MSE loss
        
        Compare with DDPM:
        - DDPM: Predicts noise ε
        - Flow: Predicts velocity v
        
        Both are regression problems, just different targets!
        
        ============================================================
        """
        
        batch_size = x_1.shape[0]
        
        # Step 1: Sample x_0 ~ N(0,I) (starting from noise)
        x_0 = torch.randn_like(x_1, device=self.device)
        
        # Step 2: Sample random timesteps t ~ Uniform(0,1)
        t = torch.rand(batch_size, device=self.device)
        
        # Step 3: TODO - Compute x_t from probability path
        # Hint: Use self.sample_probability_path()
        raise NotImplementedError(
            "TODO: Sample x_t from probability path!\n"
            "Hint: Use self.sample_probability_path(x_0, x_1, t)"
        )
        
        # Step 4: TODO - Compute target velocity
        # Hint: Use self.compute_target_velocity()
        raise NotImplementedError(
            "TODO: Compute target velocity u_t!\n"
            "Hint: Use self.compute_target_velocity(x_0, x_1, t)"
        )
        
        # Step 5: TODO - Predict velocity with model
        # Hint: v_pred = model(x_t, t)
        raise NotImplementedError(
            "TODO: Predict velocity with model!\n"
            "Hint: v_pred = model(x_t, t)\n"
            "The model takes (x_t, t) and outputs predicted velocity"
        )
        
        # Step 6: TODO - Compute MSE loss
        # Hint: torch.nn.functional.mse_loss()
        raise NotImplementedError(
            "TODO: Compute MSE loss between predicted and target velocity!\n"
            "Hint: loss = F.mse_loss(v_pred, target_velocity)"
        )
    
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, ...],
        method: Literal["euler", "rk45"] = "euler",
        n_steps: int = 100,
        return_trajectory: bool = False,
    ) -> torch.Tensor:
        """
        Generate samples by solving the ODE.
        
        This is the key difference from DDPM:
        - DDPM: Iterative stochastic denoising
        - Flow: Solve ODE deterministically
        
        ============================================================
        LEARNING NOTE: ODE Solving
        ============================================================
        
        We solve: dx/dt = v_θ(x,t) from t=0 to t=1
        
        Two methods:
        1. Euler (manual steps):
           - Simple: x_{t+dt} = x_t + v_θ(x_t, t) * dt
           - Fast but less accurate
           - Good for prototyping
        
        2. RK45 (adaptive):
           - Sophisticated Runge-Kutta method
           - Adaptive step size
           - More accurate, fewer function evaluations
           - Use for production
        
        Try both and compare!
        
        ============================================================
        """
        
        model.eval()
        
        # Start from noise: x_0 ~ N(0,I)
        x = torch.randn(shape, device=self.device)
        
        if method == "euler":
            return self._sample_euler(model, x, n_steps, return_trajectory)
        elif method == "rk45":
            return self._sample_rk45(model, x, return_trajectory)
        else:
            raise ValueError(f"Unknown sampling method: {method}")
    
    def _sample_euler(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        n_steps: int,
        return_trajectory: bool,
    ) -> torch.Tensor:
        """
        Sample using Euler's method.
        
        Simple first-order ODE solver:
        x_{t+dt} = x_t + v_θ(x_t, t) * dt
        """
        x = x_0
        dt = 1.0 / n_steps
        
        trajectory = [x.cpu().numpy()] if return_trajectory else None
        
        for step in range(n_steps):
            t = torch.full((x.shape[0],), step / n_steps, device=self.device)
            
            # Predict velocity
            v = model(x, t)
            
            # Euler step
            x = x + v * dt
            
            if return_trajectory:
                trajectory.append(x.cpu().numpy())
        
        if return_trajectory:
            return np.array(trajectory)
        else:
            return x
    
    def _sample_rk45(
        self,
        model: nn.Module,
        x_0: torch.Tensor,
        return_trajectory: bool,
    ) -> torch.Tensor:
        """
        Sample using adaptive RK45 solver.
        
        More sophisticated, adaptive step size.
        Generally more accurate with fewer evaluations.
        """
        
        def ode_func(t, x):
            """ODE function: dx/dt = v_θ(x, t)"""
            t_batch = torch.full((x.shape[0],), t, device=x.device)
            return model(x, t_batch)
        
        # Time points
        if return_trajectory:
            t_span = torch.linspace(0, 1, 100, device=self.device)
        else:
            t_span = torch.tensor([0.0, 1.0], device=self.device)
        
        # Solve ODE
        trajectory = odeint(ode_func, x_0, t_span, method='dopri5')
        
        if return_trajectory:
            return trajectory.cpu().numpy()
        else:
            return trajectory[-1]


# ============================================================
# HELPER: Check if TODOs are implemented
# ============================================================

def check_implementation():
    """
    Check if student has implemented the required functions.
    
    This is called before running the main tutorial.
    """
    print("Checking implementation...")
    
    # Try to instantiate
    flow = ConditionalFlowMatching(path_type="linear", device="cpu")
    
    # Check each TODO
    todos_remaining = []
    
    # Check probability path
    try:
        x = torch.randn(4, 2)
        t = torch.rand(4)
        flow.sample_probability_path(x, x, t)
    except NotImplementedError as e:
        todos_remaining.append("TODO #1: sample_probability_path()")
    
    # Check velocity
    try:
        x = torch.randn(4, 2)
        t = torch.rand(4)
        flow.compute_target_velocity(x, x, t)
    except NotImplementedError as e:
        todos_remaining.append("TODO #2: compute_target_velocity()")
    
    # Check loss
    try:
        x = torch.randn(4, 2)
        from .models import SimpleMLPDenoiser
        model = SimpleMLPDenoiser(input_dim=2)
        flow.training_loss(model, x)
    except NotImplementedError as e:
        todos_remaining.append("TODO #3: training_loss()")
    
    if todos_remaining:
        print("\n" + "="*70)
        print("ATTENTION: You have TODOs to complete!")
        print("="*70)
        for todo in todos_remaining:
            print(f"  - {todo}")
        print("\nPlease implement these functions based on your lecture notes.")
        print("Then run the tutorial again!")
        print("="*70)
        return False
    else:
        print("All TODOs completed! Ready to run.")
        return True
