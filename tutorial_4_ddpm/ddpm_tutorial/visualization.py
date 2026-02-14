"""
11;rgb:1212/1010/1a1aVisualization functions for the DDPM tutorial.

This module contains functions to visualize the diffusion process,
training progress, and generated samples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Tuple
import os
from scipy.stats import wasserstein_distance


def compute_histogram_intersection(hist1, hist2):
    """
    Compute histogram intersection (overlap) between two histograms.
    
    Returns a value between 0 and 1, where 1 means perfect overlap.
    
    Args:
        hist1: First histogram (counts)
        hist2: Second histogram (counts)
    
    Returns:
        Intersection score (0 to 1)
    """
    # Normalize histograms
    hist1_norm = hist1 / (hist1.sum() + 1e-10)
    hist2_norm = hist2 / (hist2.sum() + 1e-10)
    
    # Compute intersection
    intersection = np.minimum(hist1_norm, hist2_norm).sum()
    
    return intersection

def compute_histogram_chi2(hist1, hist2):
    """
    Compute the residual difference between the two
    histograms, normalised by the combined statistical
    error on each bin
    """
    # Normalize the histograms
    h1 = np.where(hist1 > 0, hist1, np.nan)  #/ (hist1.sum() + 1e-10)
    h2 = np.where(hist2 > 0, hist2, np.nan)  #/ (hist2.sum() + 1e-10)

    return np.sum( ((np.sum(hist2)*h1)-(np.sum(hist1)*h2))**2 / (h1+h2) ) / (np.sum(hist1)*np.sum(hist2))
    


def visualize_marginal_distributions(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    n_bins: int = 50,
    save_path: Optional[str] = None,
):
    """
    Visualize 1D marginal distributions for x1 and x2 dimensions.
    
    Shows histograms of real vs. generated data for each dimension separately,
    along with agreement metrics (Wasserstein distance and histogram intersection).
    
    Args:
        real_data: Real data samples, shape (n_samples, 2)
        generated_data: Generated samples, shape (n_samples, 2)
        n_bins: Number of bins for histograms
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    real_np = real_data.cpu().numpy()
    gen_np = generated_data.cpu().numpy()
    
    # Define consistent x-axis range for both dimensions
    x_range = (-3, 3)
    bins = np.linspace(x_range[0], x_range[1], n_bins + 1)
    
    # Dimension names
    dim_names = ['$x_1$', '$x_2$']
    
    for dim_idx, (ax, dim_name) in enumerate(zip(axes, dim_names)):
        # Extract 1D marginals
        real_marginal = real_np[:, dim_idx]
        gen_marginal = gen_np[:, dim_idx]
        
        # Compute histograms
        real_hist, _ = np.histogram(real_marginal, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_marginal, bins=bins, density=True)
        
        # Compute agreement metrics
        # 1. Wasserstein distance (Earth Mover's Distance)
        w_dist = wasserstein_distance(real_marginal, gen_marginal)
        
        # 2. Histogram intersection (overlap)
        real_hist_counts, _ = np.histogram(real_marginal, bins=bins)
        gen_hist_counts, _ = np.histogram(gen_marginal, bins=bins)
        #hist_intersection = compute_histogram_intersection(real_hist_counts, gen_hist_counts)
        hist_intersection = compute_histogram_chi2(real_hist_counts, gen_hist_counts)
        
        # Plot histograms
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]
        
        # Real data (blue)
        ax.bar(bin_centers, real_hist, width=width, alpha=0.6, 
               color='blue', label='Real', edgecolor='darkblue', linewidth=0.5)
        
        # Generated data (red)
        ax.bar(bin_centers, gen_hist, width=width, alpha=0.6,
               color='red', label='Generated', edgecolor='darkred', linewidth=0.5)
        
        # Formatting
        ax.set_xlabel(dim_name, fontsize=14)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Marginal Distribution: {dim_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_range)
        
        # Add metrics text box
        metrics_text = f'Wasserstein Dist: {w_dist:.4f}\nHistogram Overlap: {hist_intersection:.4f}'
        ax.text(0.02, 0.98, metrics_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.suptitle('1D Marginal Distributions: Real vs. Generated', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved marginal distributions to {save_path}")
    
    plt.show()


def visualize_forward_process(
    diffusion,
    x_0: torch.Tensor,
    timesteps: list = [0, 10, 20, 50, 99],
    save_path: Optional[str] = None,
):
    """
    Visualize the forward diffusion process (adding noise).
    
    Shows how clean data gradually becomes pure noise.
    
    Args:
        diffusion: GaussianDiffusion instance
        x_0: Clean data samples
        timesteps: List of timesteps to visualize
        save_path: Path to save figure
    """
    n_steps = len(timesteps)
    fig, axes = plt.subplots(1, n_steps, figsize=(4 * n_steps, 4))
    
    if n_steps == 1:
        axes = [axes]
    
    x_0_sample = x_0[:500].to(diffusion.device)  # Take subset for visualization
    
    for idx, (ax, t) in enumerate(zip(axes, timesteps)):
        # Add noise
        t_tensor = torch.full((len(x_0_sample),), t, device=diffusion.device, dtype=torch.long)
        x_t, _ = diffusion.q_sample(x_0_sample, t_tensor)
        
        # Convert to numpy
        x_t_np = x_t.cpu().numpy()
        
        # Plot
        ax.scatter(x_t_np[:, 0], x_t_np[:, 1], alpha=0.5, s=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f't = {t}')
        ax.grid(True, alpha=0.3)
        
        if idx == 0:
            ax.set_ylabel('$x_2$')
        ax.set_xlabel('$x_1$')
    
    plt.suptitle('Forward Diffusion Process: Clean Data → Noise', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved forward process visualization to {save_path}")
    
    plt.show()


def visualize_samples(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Compare real vs. generated samples side by side.
    
    Args:
        real_data: Real data samples
        generated_data: Generated samples
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real data
    real_np = real_data.cpu().numpy()
    axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].set_title('Real Data', fontsize=14)
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].grid(True, alpha=0.3)
    
    # Generated data
    gen_np = generated_data.cpu().numpy()
    axes[1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Generated Data', fontsize=14)
    axes[1].set_xlabel('$x_1$')
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('Real vs. Generated Samples', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved sample comparison to {save_path}")
    
    plt.show()


def plot_training_curves(
    losses: list,
    save_path: Optional[str] = None,
):
    """
    Plot training loss curves.
    
    Args:
        losses: List of loss values per epoch
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.plot(losses, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curve to {save_path}")
    
    plt.show()


def visualize_reverse_process_trajectory(
    trajectory: np.ndarray,
    n_frames_to_show: int = 10,
    save_path: Optional[str] = None,
):
    """
    Visualize the reverse sampling trajectory.
    
    Shows multiple frames of the denoising process.
    
    Args:
        trajectory: Array of shape (n_steps, batch_size, 2)
        n_frames_to_show: Number of frames to display
        save_path: Path to save figure
    """
    n_steps = len(trajectory)
    indices = np.linspace(0, n_steps - 1, n_frames_to_show, dtype=int)
    
    n_cols = 5
    n_rows = (n_frames_to_show + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = axes.flatten()
    
    for idx, frame_idx in enumerate(indices):
        ax = axes[idx]
        data = trajectory[frame_idx]
        
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=10)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'Step {n_steps - frame_idx - 1}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_frames_to_show, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Reverse Sampling Process: Noise → Data', fontsize=16, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reverse process visualization to {save_path}")
    
    plt.show()


def create_reverse_process_animation(
    trajectory: np.ndarray,
    save_path: str = "reverse_sampling.gif",
    fps: int = 20,
):
    """
    Create an animation of the reverse sampling process.
    
    Args:
        trajectory: Array of shape (n_steps, batch_size, 2)
        save_path: Path to save animation
        fps: Frames per second
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    n_steps = len(trajectory)
    
    def init():
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        return []
    
    def update(frame):
        ax.clear()
        data = trajectory[frame]
        
        ax.scatter(data[:, 0], data[:, 1], alpha=0.5, s=20, c='red')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f'Reverse Sampling: Step {n_steps - frame}/{n_steps}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        
        return []
    
    anim = FuncAnimation(
        fig, update, frames=n_steps, init_func=init,
        blit=False, repeat=True, interval=1000//fps
    )
    
    writer = PillowWriter(fps=fps)
    anim.save(save_path, writer=writer)
    print(f"Saved reverse sampling animation to {save_path}")
    plt.close()


def visualize_noise_schedule(
    diffusion,
    save_path: Optional[str] = None,
):
    """
    Visualize the noise schedule (betas, alphas, alpha_bars).
    
    Args:
        diffusion: GaussianDiffusion instance
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    timesteps = np.arange(diffusion.n_steps)
    
    # Betas
    axes[0].plot(timesteps, diffusion.betas.cpu().numpy(), linewidth=2)
    axes[0].set_xlabel('Timestep t', fontsize=12)
    axes[0].set_ylabel('$\\beta_t$', fontsize=12)
    axes[0].set_title('Variance Schedule', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # Alphas
    axes[1].plot(timesteps, diffusion.alphas.cpu().numpy(), linewidth=2)
    axes[1].set_xlabel('Timestep t', fontsize=12)
    axes[1].set_ylabel('$\\alpha_t = 1 - \\beta_t$', fontsize=12)
    axes[1].set_title('Alpha Values', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    # Cumulative alphas
    axes[2].plot(timesteps, diffusion.alpha_bars.cpu().numpy(), linewidth=2)
    axes[2].set_xlabel('Timestep t', fontsize=12)
    axes[2].set_ylabel('$\\bar{\\alpha}_t = \\prod_{s=1}^t \\alpha_s$', fontsize=12)
    axes[2].set_title('Cumulative Alpha (Signal)', fontsize=14)
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved noise schedule visualization to {save_path}")
    
    plt.show()


def create_summary_figure(
    real_data: torch.Tensor,
    generated_data: torch.Tensor,
    losses: list,
    diffusion,
    save_path: str = "ddpm_summary.png",
):
    """
    Create a comprehensive summary figure with all key visualizations.
    
    Args:
        real_data: Real data samples
        generated_data: Generated samples
        losses: Training losses
        diffusion: GaussianDiffusion instance
        save_path: Path to save figure
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Real vs Generated
    ax1 = fig.add_subplot(gs[0, 0])
    real_np = real_data.cpu().numpy()
    ax1.scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=10, c='blue', label='Real')
    ax1.set_xlim(-3, 3)
    ax1.set_ylim(-3, 3)
    ax1.set_aspect('equal')
    ax1.set_title('Real Data', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = fig.add_subplot(gs[0, 1])
    gen_np = generated_data.cpu().numpy()
    ax2.scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, s=10, c='red', label='Generated')
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-3, 3)
    ax2.set_aspect('equal')
    ax2.set_title('Generated Data', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Training curve
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(losses, linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Loss', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Noise schedule
    ax4 = fig.add_subplot(gs[1, :])
    timesteps = np.arange(diffusion.n_steps)
    ax4.plot(timesteps, diffusion.betas.cpu().numpy(), label='$\\beta_t$', linewidth=2)
    ax4.plot(timesteps, diffusion.alphas.cpu().numpy(), label='$\\alpha_t$', linewidth=2)
    ax4.plot(timesteps, diffusion.alpha_bars.cpu().numpy(), label='$\\bar{\\alpha}_t$', linewidth=2)
    ax4.set_xlabel('Timestep t')
    ax4.set_ylabel('Value')
    ax4.set_title('Noise Schedule', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('DDPM Tutorial Summary', fontsize=16, fontweight='bold')
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary figure to {save_path}")
    plt.close()
