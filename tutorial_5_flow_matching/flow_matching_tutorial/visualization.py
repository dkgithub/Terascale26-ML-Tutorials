"""
Visualization functions for the Flow Matching tutorial.

This module contains functions to visualize the flow matching process,
probability paths, velocity fields, and comparisons with DDPM.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Tuple
import os
from scipy.stats import wasserstein_distance


def visualize_probability_paths(
    flow,
    save_path: Optional[str] = None,
):
    """
    Visualize different probability paths between two points.
    
    Shows how different interpolation schemes create different trajectories.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Use points that clearly show the difference!
    # NOT symmetric or opposite - this creates visible path differences
    x_0 = torch.tensor([[0.0, -2.0]], dtype=torch.float32, device=flow.device)
    x_1 = torch.tensor([[2.0, 1.0]], dtype=torch.float32, device=flow.device)
    
    # Sample along path
    ts = torch.linspace(0, 1, 100, device=flow.device)
    
    # Get original path type
    original_path = flow.path_type
    
    # Plot both path types
    for idx, (ax, path_type) in enumerate(zip(axes, ["linear", "variance_preserving"])):
        flow.path_type = path_type
        
        path = []
        for t in ts:
            x_t = flow.sample_probability_path(
                x_0, x_1, torch.tensor([t], dtype=torch.float32, device=flow.device)
            )
            path.append(x_t[0].cpu().numpy())
        
        path = np.array(path)
        
        # Plot path
        ax.plot(path[:, 0], path[:, 1], 'b-', linewidth=2, label=f'{path_type} path')
        ax.scatter([x_0[0, 0].cpu()], [x_0[0, 1].cpu()], c='green', s=150, 
                   label='Start (noise)', zorder=5, marker='o')
        ax.scatter([x_1[0, 0].cpu()], [x_1[0, 1].cpu()], c='red', s=150,
                   label='End (data)', zorder=5, marker='s')
        
        # Add arrows to show direction
        n_arrows = 8
        arrow_indices = np.linspace(5, len(path)-5, n_arrows, dtype=int)
        for i in arrow_indices:
            dx = path[i+1, 0] - path[i, 0]
            dy = path[i+1, 1] - path[i, 1]
            ax.arrow(path[i, 0], path[i, 1], dx, dy, 
                    head_width=0.15, head_length=0.1, fc='blue', ec='blue', alpha=0.5)
        
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'Probability Path: {path_type}', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Restore original path type
    flow.path_type = original_path
    
    plt.suptitle('Probability Paths: Linear vs. Variance Preserving', fontsize=15, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved probability paths visualization to {save_path}")
    
    plt.show()


def visualize_velocity_field(
    model: torch.nn.Module,
    flow,
    t_value: float = 0.5,
    save_path: Optional[str] = None,
):
    """
    Visualize the learned velocity field at a specific time.
    """
    model.eval()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create grid
    x = np.linspace(-3, 3, 20)
    y = np.linspace(-3, 3, 20)
    X, Y = np.meshgrid(x, y)
    
    # Compute velocities
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    
    with torch.no_grad():
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pos = torch.tensor([[X[i,j], Y[i,j]]], dtype=torch.float32, device=flow.device)
                t = torch.tensor([t_value], dtype=torch.float32, device=flow.device)
                vel = model(pos, t)
                U[i,j] = vel[0, 0].cpu().numpy()
                V[i,j] = vel[0, 1].cpu().numpy()
    
    # Plot velocity field
    ax.quiver(X, Y, U, V, alpha=0.7, color='blue')
    
    ax.set_xlabel('$x_1$', fontsize=12)
    ax.set_ylabel('$x_2$', fontsize=12)
    ax.set_title(f'Velocity Field at t = {t_value}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved velocity field visualization to {save_path}")
    
    plt.show()


def compare_ode_solvers(
    samples_euler: torch.Tensor,
    samples_rk45: torch.Tensor,
    real_data,  # Can be Tensor or numpy array
    nfe_euler: int,
    nfe_rk45: int,
    save_path: Optional[str] = None,
):
    """
    Compare samples from different ODE solvers.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Real data - handle both Tensor and numpy array
    if torch.is_tensor(real_data):
        real_np = real_data.cpu().numpy()
    else:
        real_np = real_data
    axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].set_title('Real Data', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].grid(True, alpha=0.3)
    
    # Euler samples - handle both Tensor and numpy array
    if torch.is_tensor(samples_euler):
        euler_np = samples_euler.cpu().numpy()
    else:
        euler_np = samples_euler
    axes[1].scatter(euler_np[:, 0], euler_np[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title(f'Euler (NFE={nfe_euler})', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('$x_1$')
    axes[1].grid(True, alpha=0.3)
    
    # RK45 samples - handle both Tensor and numpy array
    if torch.is_tensor(samples_rk45):
        rk45_np = samples_rk45.cpu().numpy()
    else:
        rk45_np = samples_rk45
    axes[2].scatter(rk45_np[:, 0], rk45_np[:, 1], alpha=0.5, s=10, c='green')
    axes[2].set_xlim(-3, 3)
    axes[2].set_ylim(-3, 3)
    axes[2].set_aspect('equal')
    axes[2].set_title(f'RK45 (NFE={nfe_rk45})', fontsize=13, fontweight='bold')
    axes[2].set_xlabel('$x_1$')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('ODE Solver Comparison', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved ODE solver comparison to {save_path}")
    
    plt.show()


def compute_histogram_intersection(hist1, hist2):
    """
    Compute histogram intersection (overlap) between two histograms.
    """
    # Normalize histograms
    hist1_norm = hist1 / (hist1.sum() + 1e-10)
    hist2_norm = hist2 / (hist2.sum() + 1e-10)
    
    # Compute intersection
    intersection = np.minimum(hist1_norm, hist2_norm).sum()
    
    return intersection


def visualize_marginal_distributions(
    real_data,  # Can be Tensor or numpy array
    generated_data,  # Can be Tensor or numpy array
    n_bins: int = 50,
    save_path: Optional[str] = None,
):
    """
    Visualize 1D marginal distributions for x1 and x2 dimensions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to numpy - handle both Tensor and numpy array
    if torch.is_tensor(real_data):
        real_np = real_data.cpu().numpy()
    else:
        real_np = real_data
    
    if torch.is_tensor(generated_data):
        gen_np = generated_data.cpu().numpy()
    else:
        gen_np = generated_data
    
    # Define consistent x-axis range
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
        w_dist = wasserstein_distance(real_marginal, gen_marginal)
        
        # Histogram intersection
        real_hist_counts, _ = np.histogram(real_marginal, bins=bins)
        gen_hist_counts, _ = np.histogram(gen_marginal, bins=bins)
        hist_intersection = compute_histogram_intersection(real_hist_counts, gen_hist_counts)
        
        # Plot histograms
        bin_centers = (bins[:-1] + bins[1:]) / 2
        width = bins[1] - bins[0]
        
        ax.bar(bin_centers, real_hist, width=width, alpha=0.6, 
               color='blue', label='Real', edgecolor='darkblue', linewidth=0.5)
        
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


def visualize_samples(
    real_data,  # Can be Tensor or numpy array
    generated_data,  # Can be Tensor or numpy array
    save_path: Optional[str] = None,
):
    """
    Visualize real vs generated samples side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Real data - handle both Tensor and numpy array
    if torch.is_tensor(real_data):
        real_np = real_data.cpu().numpy()
    else:
        real_np = real_data
    axes[0].scatter(real_np[:, 0], real_np[:, 1], alpha=0.5, s=10, c='blue')
    axes[0].set_xlim(-3, 3)
    axes[0].set_ylim(-3, 3)
    axes[0].set_aspect('equal')
    axes[0].set_title('Real Data', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('$x_1$')
    axes[0].set_ylabel('$x_2$')
    axes[0].grid(True, alpha=0.3)
    
    # Generated data - handle both Tensor and numpy array
    if torch.is_tensor(generated_data):
        gen_np = generated_data.cpu().numpy()
    else:
        gen_np = generated_data
    axes[1].scatter(gen_np[:, 0], gen_np[:, 1], alpha=0.5, s=10, c='red')
    axes[1].set_xlim(-3, 3)
    axes[1].set_ylim(-3, 3)
    axes[1].set_aspect('equal')
    axes[1].set_title('Generated Data', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('$x_1$')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples comparison to {save_path}")
    
    plt.show()


def plot_training_curves(
    losses: list,
    save_path: Optional[str] = None,
):
    """
    Plot training loss curve.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(losses, linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
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
    Visualize the reverse process trajectory (ODE integration).
    """
    n_steps = len(trajectory)
    indices = np.linspace(0, n_steps-1, n_frames_to_show, dtype=int)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, step_idx in enumerate(indices):
        ax = axes[idx]
        samples = trajectory[step_idx]
        
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=5)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_title(f't = {step_idx/n_steps:.2f}', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        if idx >= 5:
            ax.set_xlabel('$x_1$')
        if idx % 5 == 0:
            ax.set_ylabel('$x_2$')
    
    plt.suptitle('Flow Matching: ODE Integration Process', fontsize=16, y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {save_path}")
    
    plt.show()


def create_reverse_process_animation(
    trajectory: np.ndarray,
    save_path: Optional[str] = None,
    fps: int = 20,
):
    """
    Create animated GIF of the reverse process.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        ax.clear()
        samples = trajectory[frame]
        ax.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10, c='blue')
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_aspect('equal')
        ax.set_xlabel('$x_1$', fontsize=12)
        ax.set_ylabel('$x_2$', fontsize=12)
        ax.set_title(f'Flow Matching Sampling (t = {frame/len(trajectory):.2f})', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, update, frames=len(trajectory), interval=1000//fps)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")
    
    plt.close()


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
