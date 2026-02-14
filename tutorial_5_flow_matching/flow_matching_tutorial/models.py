"""
Neural network architectures for the DDPM tutorial.

This module contains simple MLP architectures for predicting noise
in the diffusion process.
"""

import torch
import torch.nn as nn
import numpy as np


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal position embedding for timesteps.
    
    This helps the network understand "when" in the diffusion process
    we are (what timestep t).
    
    Reference: "Attention is All You Need" (Transformer paper)
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Timesteps, shape (batch_size,)
        
        Returns:
            Embeddings, shape (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class SimpleMLPDenoiser(nn.Module):
    """
    Simple Multi-Layer Perceptron for predicting noise.
    
    Architecture:
        Input: [x_t, t] concatenated
        Hidden layers: Multiple fully-connected layers with activation
        Output: Predicted noise ε_θ(x_t, t)
    
    The goal: Given a noisy sample x_t and timestep t, predict the noise
    that was added to create x_t from x_0.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        n_layers: int = 3,
    ):
        """
        Args:
            input_dim: Dimension of input data (2 for 2D data)
            hidden_dim: Size of hidden layers
            time_embed_dim: Dimension of time embeddings
            n_layers: Number of hidden layers
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.time_embed_dim = time_embed_dim
        
        # Time embedding layer
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(time_embed_dim * 2, time_embed_dim),
        )
        
        # Main network: processes [x_t, time_embedding]
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim + time_embed_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer: predict noise (same dim as input)
        layers.append(nn.Linear(hidden_dim, input_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: predict noise.
        
        Args:
            x: Noisy samples, shape (batch_size, input_dim)
            t: Timesteps, shape (batch_size,)
        
        Returns:
            Predicted noise, shape (batch_size, input_dim)
        """
        # Embed timesteps
        t_embed = self.time_mlp(t)  # (batch_size, time_embed_dim)
        
        # Concatenate x and time embedding
        x_input = torch.cat([x, t_embed], dim=-1)  # (batch_size, input_dim + time_embed_dim)
        
        # Predict noise
        noise_pred = self.network(x_input)
        
        return noise_pred


class ResidualBlock(nn.Module):
    """
    Residual block for deeper networks.
    
    This is optional and can improve performance for more complex datasets.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.network(x)


class ImprovedMLPDenoiser(nn.Module):
    """
    Improved MLP with residual connections.
    
    For students who want to experiment with better architectures.
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        n_residual_blocks: int = 3,
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim * 4),
            nn.ReLU(),
            nn.Linear(time_embed_dim * 4, time_embed_dim),
        )
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_residual_blocks)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_embed = self.time_mlp(t)
        x_input = torch.cat([x, t_embed], dim=-1)
        
        h = self.input_layer(x_input)
        
        for block in self.residual_blocks:
            h = block(h)
        
        return self.output_layer(h)


# For students to experiment
def get_model(model_type: str = "simple", **kwargs):
    """
    Factory function to get different model architectures.
    
    Args:
        model_type: "simple" or "improved"
        **kwargs: Model-specific parameters
    
    Returns:
        Model instance
    """
    if model_type == "simple":
        return SimpleMLPDenoiser(**kwargs)
    elif model_type == "improved":
        return ImprovedMLPDenoiser(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
