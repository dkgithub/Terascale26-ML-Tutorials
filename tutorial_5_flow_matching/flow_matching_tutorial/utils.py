"""
Utility functions for the DDPM tutorial.

This includes toy dataset generation and helper functions.
"""

import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles, make_swiss_roll
from typing import Tuple, Literal


def create_toy_dataset(
    dataset_type: Literal["moons", "circles", "swiss_roll", "two_gaussians", "checkerboard"] = "moons",
    n_samples: int = 10000,
    noise: float = 0.05,
) -> torch.Tensor:
    """
    Create 2D toy datasets for visualization.
    
    Args:
        dataset_type: Type of dataset
        n_samples: Number of samples to generate
        noise: Noise level for the dataset
    
    Returns:
        Samples as torch tensor, shape (n_samples, 2)
    """
    
    if dataset_type == "moons":
        # Two interleaving half-circles
        data, _ = make_moons(n_samples=n_samples, noise=noise)
        data = data.astype(np.float32)
        # Normalize to roughly [-2, 2] range
        data = (data - data.mean(axis=0)) / data.std(axis=0) * 1.5
    
    elif dataset_type == "circles":
        # Two concentric circles
        data, _ = make_circles(n_samples=n_samples, noise=noise, factor=0.5)
        data = data.astype(np.float32)
        data = (data - data.mean(axis=0)) / data.std(axis=0) * 1.5
    
    elif dataset_type == "swiss_roll":
        # Classic swiss roll (2D projection)
        data, _ = make_swiss_roll(n_samples=n_samples, noise=noise)
        # Take only x and z coordinates
        data = data[:, [0, 2]].astype(np.float32)
        data = (data - data.mean(axis=0)) / data.std(axis=0) * 1.5
    
    elif dataset_type == "two_gaussians":
        # Two well-separated Gaussians
        n_per_gaussian = n_samples // 2
        
        # First Gaussian
        gaussian1 = np.random.randn(n_per_gaussian, 2).astype(np.float32) * 0.5
        gaussian1[:, 0] += 1.5
        
        # Second Gaussian
        gaussian2 = np.random.randn(n_samples - n_per_gaussian, 2).astype(np.float32) * 0.5
        gaussian2[:, 0] -= 1.5
        
        data = np.vstack([gaussian1, gaussian2])
        np.random.shuffle(data)
    
    elif dataset_type == "checkerboard":
        # Checkerboard pattern (challenging!)
        x = np.random.uniform(-4, 4, n_samples)
        y = np.random.uniform(-4, 4, n_samples)
        
        # Create checkerboard mask
        mask = ((x // 2) % 2 == 0) == ((y // 2) % 2 == 0)
        
        data = np.stack([x[mask], y[mask]], axis=1).astype(np.float32)
        
        # Add noise
        data += np.random.randn(*data.shape).astype(np.float32) * noise
        
        # Take only n_samples
        if len(data) > n_samples:
            data = data[:n_samples]
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return torch.from_numpy(data)


def compute_dataset_stats(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and std of dataset.
    
    Args:
        data: Dataset tensor
    
    Returns:
        mean, std
    """
    mean = data.mean(dim=0)
    std = data.std(dim=0)
    return mean, std


class ToyDataLoader:
    """
    Simple data loader for toy datasets.
    
    This is simpler than PyTorch's DataLoader and sufficient for our needs.
    """
    
    def __init__(self, data: torch.Tensor, batch_size: int, shuffle: bool = True):
        self.data = data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(data)
        self.n_batches = (self.n_samples + batch_size - 1) // batch_size
    
    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(self.n_samples)
            data = self.data[indices]
        else:
            data = self.data
        
        for i in range(0, self.n_samples, self.batch_size):
            batch = data[i:i + self.batch_size]
            yield batch
    
    def __len__(self):
        return self.n_batches


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device() -> str:
    """
    Get the best available device.
    
    Returns:
        Device string ("cuda" or "cpu")
    """
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str,
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        path: Path to save checkpoint
    """
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    path: str,
    device: str,
) -> Tuple[int, float]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        path: Path to checkpoint
        device: Device to load to
    
    Returns:
        epoch, loss
    """
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"], checkpoint["loss"]
