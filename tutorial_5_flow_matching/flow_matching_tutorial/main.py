"""
Tutorial 2: Flow Matching - From SDEs to ODEs

A complete walkthrough of flow matching for generative modeling.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
import time

from .models import SimpleMLPDenoiser
from .flow import ConditionalFlowMatching, check_implementation
from .utils import (
    create_toy_dataset,
    ToyDataLoader,
    set_seed,
    count_parameters,
    get_device,
)
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


def main():
    """Main tutorial function."""
    
    print("=" * 70)
    print("Tutorial 2: Flow Matching - From SDEs to ODEs")
    print("=" * 70)
    print()
    
    # Check implementation
    print("Step 0: Checking Your Implementation")
    print("-" * 70)
    
    if not check_implementation():
        print("\nPlease complete the TODOs in flow.py!")
        return
    
    print()
    
    # Setup
    print("Step 1: Setup")
    print("-" * 70)
    
    set_seed(42)
    device = get_device()
    print(f"Using device: {device}")
    
    os.makedirs("outputs", exist_ok=True)
    
    config = {
        "dataset_type": "moons",
        "n_samples": 10000,
        "noise": 0.05,
        "path_type": "variance_preserving",
        "hidden_dim": 128,
        "time_embed_dim": 32,
        "n_layers": 3,
        "batch_size": 256,
        "n_epochs": 100,
        "learning_rate": 1e-3,
        "n_samples_to_generate": 1000,
        "n_euler_steps": 100,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Create dataset
    print("Step 2: Creating Dataset")
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
    
    data_loader = ToyDataLoader(
        data, batch_size=config["batch_size"], shuffle=True
    )
    
    # Initialize flow matching
    print("Step 3: Initializing Flow Matching")
    print("-" * 70)
    
    flow = ConditionalFlowMatching(
        path_type=config["path_type"],
        device=device,
    )
    
    print(f"Created Conditional Flow Matching")
    print(f"  Probability path: {config['path_type']}")
    print(f"  Process type: ODE (deterministic)")
    print()
    
    print("Visualizing probability paths...")
    visualize_probability_paths(flow, save_path="outputs/probability_paths.png")
    
    # Create model
    print("Step 4: Creating Model")
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
    print(f"  Total parameters: {n_params:,}")
    print()
    
    # Train
    print("Step 5: Training")
    print("-" * 70)
    print("Target: Predict velocity v(x,t)")
    print()
    
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    losses = []
    
    model.train()
    
    for epoch in range(config['n_epochs']):
        epoch_losses = []
        
        pbar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")
        
        for batch in pbar:
            batch = batch.to(device)
            
            loss = flow.training_loss(model, batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(avg_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['n_epochs']}, Loss: {avg_loss:.4f}")
    
    print(f"\nTraining complete! Final loss: {losses[-1]:.4f}")
    print()
    
    plot_training_curves(losses, save_path="outputs/training_curve.png")
    
    print("Visualizing velocity field...")
    visualize_velocity_field(model, flow, t_value=0.5, save_path="outputs/velocity_field.png")
    
    # Generate samples
    print("Step 6: Generating Samples")
    print("-" * 70)
    
    model.eval()
    
    print(f"Method 1: Euler solver ({config['n_euler_steps']} steps)")
    start_time = time.time()
    
    trajectory_euler = flow.sample(
        model,
        shape=(config["n_samples_to_generate"], 2),
        method="euler",
        n_steps=config["n_euler_steps"],
        return_trajectory=True,
    )
    
    euler_time = time.time() - start_time
    samples_euler = torch.from_numpy(trajectory_euler[-1])
    print(f"  Sampling time: {euler_time:.2f}s")
    
    print(f"\nMethod 2: RK45 solver (adaptive)")
    start_time = time.time()
    
    samples_rk45 = flow.sample(
        model,
        shape=(config["n_samples_to_generate"], 2),
        method="rk45",
        return_trajectory=False,
    )
    
    rk45_time = time.time() - start_time
    print(f"  Sampling time: {rk45_time:.2f}s")
    print()
    
    print("Comparing ODE solvers...")
    compare_ode_solvers(
        samples_euler,
        samples_rk45,
        data[:config["n_samples_to_generate"]],
        nfe_euler=config["n_euler_steps"],
        nfe_rk45=30,
        save_path="outputs/ode_solver_comparison.png"
    )
    
    print("Visualizing trajectory...")
    visualize_reverse_process_trajectory(
        trajectory_euler,
        n_frames_to_show=10,
        save_path="outputs/flow_trajectory.png"
    )
    
    print("Creating animation...")
    create_reverse_process_animation(
        trajectory_euler,
        save_path="outputs/flow_sampling.gif",
        fps=20,
    )
    
    # Compare
    print("Step 7: Comparing Results")
    print("-" * 70)
    
    visualize_samples(
        data[:config["n_samples_to_generate"]],
        samples_rk45,
        save_path="outputs/real_vs_generated.png"
    )
    
    visualize_marginal_distributions(
        data[:config["n_samples_to_generate"]],
        samples_rk45,
        save_path="outputs/marginal_distributions.png"
    )
    
    print()
    print("=" * 70)
    print("Tutorial Complete!")
    print("=" * 70)
    print()
    print("Generated files in outputs/:")
    print("  • probability_paths.png")
    print("  • velocity_field.png")
    print("  • training_curve.png")
    print("  • ode_solver_comparison.png")
    print("  • flow_trajectory.png")
    print("  • flow_sampling.gif")
    print("  • real_vs_generated.png")
    print("  • marginal_distributions.png")
    print()
    print("Key Insights:")
    print("  1. Flow Matching uses ODEs (deterministic)")
    print("  2. Same model - only target changes (velocity vs noise)")
    print("  3. Probability path choice matters")
    print("  4. ODE solver selection affects speed/quality")
    print()


if __name__ == "__main__":
    main()
