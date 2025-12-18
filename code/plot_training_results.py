#!/usr/bin/env python3
"""
Plot training results for all model variants.
Compares rewards and losses across base, dueling, mha, and dueling_mha models.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CHECKPOINTS_DIR = "checkpoints_ec2"
PLOTS_DIR = "plots"
MODELS = ["base", "dueling", "mha", "dueling_mha"]
MODEL_LABELS = {
    "base": "Base DQN",
    "dueling": "Dueling DQN",
    "mha": "MHA DQN",
    "dueling_mha": "Dueling + MHA DQN"
}
COLORS = {
    "base": "#1f77b4",      # Blue
    "dueling": "#ff7f0e",   # Orange
    "mha": "#2ca02c",       # Green
    "dueling_mha": "#d62728" # Red
}


def load_training_logs(checkpoints_dir: str) -> dict:
    """Load all training logs from the checkpoints directory."""
    logs = {}
    for model in MODELS:
        log_path = Path(checkpoints_dir) / f"{model}_logs.json"
        if log_path.exists():
            with open(log_path, "r") as f:
                logs[model] = json.load(f)
            print(f"✓ Loaded {model} logs: {len(logs[model]['episode_rewards'])} episodes")
        else:
            print(f"✗ Missing {model} logs")
    return logs


def smooth_curve(values: list, window: int = 100) -> np.ndarray:
    """Apply moving average smoothing to a curve."""
    values = np.array(values)
    if len(values) < window:
        window = max(1, len(values) // 5)
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    return smoothed


def plot_episode_rewards(logs: dict, output_dir: str):
    """Plot episode rewards for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw rewards
    ax1 = axes[0]
    for model, data in logs.items():
        rewards = data["episode_rewards"]
        episodes = range(1, len(rewards) + 1)
        ax1.plot(episodes, rewards, alpha=0.3, color=COLORS[model])
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Episode Reward", fontsize=12)
    ax1.set_title("Raw Episode Rewards", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend([MODEL_LABELS[m] for m in logs.keys()])
    
    # Smoothed rewards
    ax2 = axes[1]
    for model, data in logs.items():
        rewards = data["episode_rewards"]
        smoothed = smooth_curve(rewards, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax2.plot(episodes, smoothed, linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Episode Reward (100-ep Moving Avg)", fontsize=12)
    ax2.set_title("Smoothed Episode Rewards", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "episode_rewards.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_episode_losses(logs: dict, output_dir: str):
    """Plot episode losses for all models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw losses
    ax1 = axes[0]
    for model, data in logs.items():
        losses = data["episode_losses"]
        episodes = range(1, len(losses) + 1)
        ax1.plot(episodes, losses, alpha=0.3, color=COLORS[model])
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Episode Loss", fontsize=12)
    ax1.set_title("Raw Episode Losses", fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend([MODEL_LABELS[m] for m in logs.keys()])
    
    # Smoothed losses
    ax2 = axes[1]
    for model, data in logs.items():
        losses = data["episode_losses"]
        smoothed = smooth_curve(losses, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax2.plot(episodes, smoothed, linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Episode Loss (100-ep Moving Avg)", fontsize=12)
    ax2.set_title("Smoothed Episode Losses", fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = Path(output_dir) / "episode_losses.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_combined_comparison(logs: dict, output_dir: str):
    """Create a comprehensive comparison plot."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Smoothed rewards comparison
    ax1 = axes[0, 0]
    for model, data in logs.items():
        rewards = data["episode_rewards"]
        smoothed = smooth_curve(rewards, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax1.plot(episodes, smoothed, linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward (100-ep Moving Avg)", fontsize=12)
    ax1.set_title("Learning Curves Comparison", fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Smoothed losses comparison
    ax2 = axes[0, 1]
    for model, data in logs.items():
        losses = data["episode_losses"]
        smoothed = smooth_curve(losses, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax2.plot(episodes, smoothed, linewidth=2, color=COLORS[model], label=MODEL_LABELS[model])
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Loss (100-ep Moving Avg)", fontsize=12)
    ax2.set_title("Loss Curves Comparison", fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Final performance bar chart
    ax3 = axes[1, 0]
    model_names = []
    avg_rewards = []
    best_rewards = []
    for model, data in logs.items():
        model_names.append(MODEL_LABELS[model])
        avg_rewards.append(data["avg_reward"])
        best_rewards.append(data["best_reward"])
    
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax3.bar(x - width/2, avg_rewards, width, label='Avg Reward', color='steelblue')
    bars2 = ax3.bar(x + width/2, best_rewards, width, label='Best Reward', color='coral')
    ax3.set_ylabel('Reward', fontsize=12)
    ax3.set_title('Final Performance Comparison', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=15)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Training time comparison
    ax4 = axes[1, 1]
    training_times = []
    for model, data in logs.items():
        training_times.append(data["total_training_time_seconds"] / 3600)  # Convert to hours
    
    bars = ax4.bar(model_names, training_times, color=[COLORS[m] for m in logs.keys()])
    ax4.set_ylabel('Training Time (hours)', fontsize=12)
    ax4.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax4.set_xticklabels(model_names, rotation=15)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, time in zip(bars, training_times):
        ax4.annotate(f'{time:.1f}h', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('DQN Model Variants Comparison - Atari Breakout', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(output_dir) / "combined_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_individual_rewards(logs: dict, output_dir: str):
    """Plot individual reward curves for each model."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for idx, (model, data) in enumerate(logs.items()):
        ax = axes[idx]
        rewards = data["episode_rewards"]
        smoothed = smooth_curve(rewards, window=50)
        
        # Plot raw rewards
        ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.2, color=COLORS[model], label='Raw')
        # Plot smoothed rewards
        ax.plot(range(50, 50 + len(smoothed)), smoothed, linewidth=2, color=COLORS[model], label='Smoothed (50-ep)')
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.set_title(f"{MODEL_LABELS[model]}\nAvg: {data['avg_reward']:.1f}, Best: {data['best_reward']:.0f}", fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
    
    plt.suptitle('Individual Model Learning Curves', fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    output_path = Path(output_dir) / "individual_rewards.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_summary(logs: dict):
    """Print a summary of training results."""
    print("\n" + "="*70)
    print("TRAINING RESULTS SUMMARY")
    print("="*70)
    print(f"{'Model':<20} {'Episodes':<10} {'Avg Reward':<12} {'Best Reward':<12} {'Training Time':<15}")
    print("-"*70)
    
    for model, data in logs.items():
        training_time = data["total_training_time_formatted"]
        print(f"{MODEL_LABELS[model]:<20} {data['total_episodes']:<10} {data['avg_reward']:<12.2f} {data['best_reward']:<12.0f} {training_time:<15}")
    
    print("="*70 + "\n")


def main():
    """Main function to generate all plots."""
    print("\n" + "="*50)
    print("Generating Training Plots")
    print("="*50 + "\n")
    
    # Create plots directory
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load logs
    logs = load_training_logs(CHECKPOINTS_DIR)
    
    if not logs:
        print("❌ No training logs found!")
        return
    
    print("\nGenerating plots...\n")
    
    # Generate plots
    plot_episode_rewards(logs, PLOTS_DIR)
    plot_episode_losses(logs, PLOTS_DIR)
    plot_combined_comparison(logs, PLOTS_DIR)
    plot_individual_rewards(logs, PLOTS_DIR)
    
    # Print summary
    print_summary(logs)
    
    print(f"✅ All plots saved to: {PLOTS_DIR}/")
    print(f"   - episode_rewards.png")
    print(f"   - episode_losses.png")
    print(f"   - combined_comparison.png")
    print(f"   - individual_rewards.png")


if __name__ == "__main__":
    main()

