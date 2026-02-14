"""
Main tutorial script: Building Your First Diffusion Model (DDPM)

This script walks through the complete process of:
1. Creating a toy 2D dataset
2. Training a denoising diffusion model
3. Generating new samples
4. Visualizing the results

Expected runtime: ~25-30 minutes (depending on hardware)
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os

from .models import SimpleMLPDenoiser
from .diffusion import GaussianDiffusion
from .utils import (
    create_toy_dataset,
    ToyDataLoader,
    set_seed,
    count_parameters,
    get_device,
)
from .visualization import (
    visualize_forward_process,
    visualize_samples,
    plot_training_curves,
    visualize_reverse_process_trajectory,
    create_reverse_process_animation,
    visualize_noise_schedule,
    create_summary_figure,
    visualize_marginal_distributions,
)


def main():
    """
    Main tutorial function.
    
    This is the complete walkthrough that students will follow.
    """
    
    print("=" * 70)
    print("Tutorial 1: Building Your First Diffusion Model (DDPM)")
    print("=" * 70)
    print()
    
    # ========================================================================
    # STEP 1: Setup and Configuration
    # ========================================================================
    print("Step 1: Setup and Configuration")
    print("-" * 70)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs("outputs", exist_ok=True)
    print("Created outputs/ directory for saving figures")
    
    # Configuration
    config = {
        # Data
        "dataset_type": "moons",  # Options: "moons", "circles", "swiss_roll", "two_gaussians"
        "n_samples": 50000,
        "noise": 0.05,
        
        # Diffusion
        "n_steps": 1000,  # Number of diffusion timesteps T
        "beta_start": 1e-5,
        "beta_end": 0.02,
        "schedule": "cosine",  # Options: "linear", "cosine"
        
        # Model
        "hidden_dim": 128,
        "time_embed_dim": 32,
        "n_layers": 3,
        
        # Training
        "batch_size": 256,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        
        # Sampling
        "n_samples_to_generate": 100000,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # ========================================================================
    # STEP 2: Create Dataset
    # ========================================================================
    print("Step 2: Creating Toy Dataset")
    print("-" * 70)
    
    data = create_toy_dataset(
        dataset_type=config["dataset_type"],
        n_samples=config["n_samples"],
        noise=config["noise"],
    )
    
    print(f"Created '{config['dataset_type']}' dataset")
    print(f"  Shape: {data.shape}")
    print(f"  Mean: [{data.mean(0)[0]:.3f}, {data.mean(0)[1]:.3f}]")
    print(f"  Std:  [{data.std(0)[0]:.3f}, {data.std(0)[1]:.3f}]")
    print()
    
    # Create data loader
    data_loader = ToyDataLoader(
        data, batch_size=config["batch_size"], shuffle=True
    )
    
    # ========================================================================
    # STEP 3: Initialize Diffusion Process
    # ========================================================================
    print("Step 3: Initializing Diffusion Process")
    print("-" * 70)
    
    diffusion = GaussianDiffusion(
        n_steps=config["n_steps"],
        beta_start=config["beta_start"],
        beta_end=config["beta_end"],
        schedule=config["schedule"],
        device=device,
    )
    
    print(f"Created Gaussian Diffusion with {config['n_steps']} steps")
    print(f"  Noise schedule: {config['schedule']}")
    print(f"  β range: [{config['beta_start']}, {config['beta_end']}]")
    print()
    
    # Visualize noise schedule
    print("Visualizing noise schedule...")
    visualize_noise_schedule(diffusion, save_path="outputs/noise_schedule.png")
    
    # Visualize forward process
    print("Visualizing forward diffusion process...")
    visualize_forward_process(
        diffusion, 
        data[:1000],
        timesteps=[0, 25, 50, 75, 99],
        save_path="outputs/forward_process.png"
    )
    
    # ========================================================================
    # STEP 4: Create and Initialize Model
    # ========================================================================
    print("Step 4: Creating Neural Network")
    print("-" * 70)
    
    model = SimpleMLPDenoiser(
        input_dim=2,
        hidden_dim=config["hidden_dim"],
        time_embed_dim=config["time_embed_dim"],
        n_layers=config["n_layers"],
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"Created SimpleMLPDenoiser")
    print(f"  Hidden dim: {config['hidden_dim']}")
    print(f"  Time embed dim: {config['time_embed_dim']}")
    print(f"  Number of layers: {config['n_layers']}")
    print(f"  Total parameters: {n_params:,}")
    print()
    
    # ========================================================================
    # STEP 5: Train the Model
    # ========================================================================
    print("Step 5: Training the Model")
    print("-" * 70)
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    losses = []
    
    print(f"Training for {config['n_epochs']} epochs...")
    print()
    
    model.train()
    
    for epoch in range(config["n_epochs"]):
        epoch_losses = []
        
        # Progress bar for batches
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            # Compute loss
            loss = diffusion.training_loss(model, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Record loss
            epoch_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        # Average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        # Print epoch summary
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']}, Loss: {avg_loss:.4f}")
    
    print()
    print("Training complete!")
    print(f"Final loss: {losses[-1]:.4f}")
    print()
    
    # Plot training curve
    print("Plotting training curve...")
    plot_training_curves(losses, save_path="outputs/training_curve.png")
    
    # ========================================================================
    # STEP 6: Generate Samples
    # ========================================================================
    print("Step 6: Generating New Samples")
    print("-" * 70)
    
    model.eval()
    
    print(f"Generating {config['n_samples_to_generate']} samples...")
    
    # Generate samples with trajectory
    trajectory = diffusion.sample(
        model,
        shape=(config["n_samples_to_generate"], 2),
        return_trajectory=True,
    )
    
    generated_samples = torch.from_numpy(trajectory[-1])
    
    print(f"Generated {len(generated_samples)} samples")
    print(f"  Shape: {generated_samples.shape}")
    print()
    
    # Visualize reverse process
    print("Visualizing reverse sampling process...")
    visualize_reverse_process_trajectory(
        trajectory,
        n_frames_to_show=10,
        save_path="outputs/reverse_process.png"
    )
    
    # Create animation
    print("Creating reverse sampling animation (this may take a moment)...")
    create_reverse_process_animation(
        trajectory,
        save_path="outputs/reverse_sampling.gif",
        fps=20,
    )
    
    # ========================================================================
    # STEP 7: Compare Real vs Generated
    # ========================================================================
    print("Step 7: Comparing Real vs Generated Samples")
    print("-" * 70)
    
    print("Creating comparison visualization...")
    visualize_samples(
        data[:config["n_samples_to_generate"]],
        generated_samples,
        save_path="outputs/real_vs_generated.png"
    )
    
    # Visualize 1D marginal distributions
    print("Creating 1D marginal distribution comparison...")
    visualize_marginal_distributions(
        data[:config["n_samples_to_generate"]],
        generated_samples,
        save_path="outputs/marginal_distributions.png"
    )
    
    # ========================================================================
    # STEP 8: Create Summary
    # ========================================================================
    print("Step 8: Creating Summary Figure")
    print("-" * 70)
    
    print("Creating comprehensive summary...")
    create_summary_figure(
        data[:config["n_samples_to_generate"]],
        generated_samples,
        losses,
        diffusion,
        save_path="outputs/ddpm_summary.png"
    )
    
    # ========================================================================
    # Tutorial Complete!
    # ========================================================================
    print()
    print("=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)
    print()
    print("Generated files in outputs/:")
    print("  • noise_schedule.png       - Visualization of β_t, α_t, ᾱ_t")
    print("  • forward_process.png      - Forward diffusion: data → noise")
    print("  • training_curve.png       - Loss curve during training")
    print("  • reverse_process.png      - Reverse sampling: noise → data")
    print("  • reverse_sampling.gif     - Animation of sampling process")
    print("  • real_vs_generated.png    - Comparison of distributions")
    print("  • marginal_distributions.png - 1D marginal distributions with metrics")
    print("  • ddpm_summary.png         - Complete summary figure")
    print()
    print("=" * 70)
    print("What You've Learned:")
    print("=" * 70)
    print(" Forward diffusion process: q(x_t | x_0)")
    print(" Training objective: E[||ε - ε_θ(x_t, t)||²]")
    print(" Reverse sampling: p_θ(x_{t-1} | x_t)")
    print(" Noise schedules and their effect")
    print()
    print("=" * 70)
    print(" Questions to Explore:")
    print("=" * 70)
    print("1. Try different datasets (change 'dataset_type')")
    print("2. Experiment with noise schedules ('linear' vs 'cosine')")
    print("3. Adjust n_steps - what happens with fewer/more steps?")
    print("4. Modify the network architecture (hidden_dim, n_layers)")
    print("5. Compare training time vs sample quality trade-offs")
    print()
    print("=" * 70)
    print(" Next Steps:")
    print("=" * 70)
    print("• Tutorial 2: Flow Matching from Scratch")
    print("• Tutorial 3: Diffusion vs Flow Matching Comparison")
    print("• Read the DDPM paper for deeper understanding")
    print()


if __name__ == "__main__":
    main()
