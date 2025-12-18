#!/usr/bin/env python3
"""
Comprehensive comparison plots for all DQN model variants.
Compares Base, Dueling, MHA, Dueling+MHA, and MHA V2 models.
"""
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
CHECKPOINTS_10K_DIR = "checkpoints_10k"
MHA_V2_DIR = "checkpoint_20k_mhaV2"
PLOTS_DIR = "plots"

MODELS_10K = ["base", "dueling", "mha", "dueling_mha"]
MODEL_LABELS = {
    "base": "Base DQN",
    "dueling": "Dueling DQN",
    "mha": "MHA DQN",
    "dueling_mha": "Dueling + MHA DQN",
    "mha_v2": "MHA V2 DQN"
}
COLORS = {
    "base": "#1f77b4",      # Blue
    "dueling": "#ff7f0e",   # Orange
    "mha": "#2ca02c",       # Green
    "dueling_mha": "#d62728", # Red
    "mha_v2": "#9467bd"     # Purple
}


def load_logs():
    """Load all training logs."""
    logs = {}
    
    # Load 10k episode models
    for model in MODELS_10K:
        log_path = Path(CHECKPOINTS_10K_DIR) / f"{model}_logs.json"
        if log_path.exists():
            with open(log_path, "r") as f:
                logs[model] = json.load(f)
            print(f"✓ Loaded {model}: {logs[model]['total_episodes']} episodes")
        else:
            print(f"✗ Missing {model} logs")
    
    # Load MHA V2
    log_path = Path(MHA_V2_DIR) / "mha_v2_logs.json"
    if log_path.exists():
        with open(log_path, "r") as f:
            logs["mha_v2"] = json.load(f)
        print(f"✓ Loaded mha_v2: {logs['mha_v2']['total_episodes']} episodes")
    else:
        print(f"✗ Missing mha_v2 logs")
    
    return logs


def smooth_curve(values: list, window: int = 100) -> np.ndarray:
    """Apply moving average smoothing."""
    values = np.array(values)
    if len(values) < window:
        window = max(1, len(values) // 5)
    weights = np.ones(window) / window
    smoothed = np.convolve(values, weights, mode='valid')
    return smoothed


def plot_comprehensive_comparison(logs: dict, output_dir: str):
    """Create comprehensive 4-panel comparison figure."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # 1. Learning Curves - Rewards (all models)
    ax1 = axes[0, 0]
    for model, data in logs.items():
        rewards = data["episode_rewards"]
        smoothed = smooth_curve(rewards, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax1.plot(episodes, smoothed, linewidth=2.5, color=COLORS[model], 
                label=MODEL_LABELS[model], alpha=0.9)
    ax1.set_xlabel("Episode", fontsize=13)
    ax1.set_ylabel("Reward (100-ep Moving Avg)", fontsize=13)
    ax1.set_title("Learning Curves - Episode Rewards", fontsize=15, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10000)
    
    # 2. Learning Curves - Losses
    ax2 = axes[0, 1]
    for model, data in logs.items():
        losses = data["episode_losses"]
        smoothed = smooth_curve(losses, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax2.plot(episodes, smoothed, linewidth=2.5, color=COLORS[model], 
                label=MODEL_LABELS[model], alpha=0.9)
    ax2.set_xlabel("Episode", fontsize=13)
    ax2.set_ylabel("Loss (100-ep Moving Avg)", fontsize=13)
    ax2.set_title("Learning Curves - Episode Losses", fontsize=15, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10000)
    
    # 3. Final Performance Bar Chart
    ax3 = axes[1, 0]
    model_names = [MODEL_LABELS[m] for m in logs.keys()]
    avg_rewards = [logs[m]["avg_reward"] for m in logs.keys()]
    best_rewards = [logs[m]["best_reward"] for m in logs.keys()]
    
    x = np.arange(len(model_names))
    width = 0.35
    bars1 = ax3.bar(x - width/2, avg_rewards, width, label='Avg Reward', 
                    color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, best_rewards, width, label='Best Reward', 
                    color='coral', alpha=0.8)
    ax3.set_ylabel('Reward', fontsize=13)
    ax3.set_title('Final Performance Comparison (10K Episodes)', fontsize=15, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=15, ha='right', fontsize=10)
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.annotate(f'{height:.0f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # 4. Training Efficiency - Episodes to reach thresholds
    ax4 = axes[1, 1]
    thresholds = [5, 10, 15, 20]
    threshold_data = {model: [] for model in logs.keys()}
    
    for model, data in logs.items():
        rewards = np.array(data["episode_rewards"])
        smoothed = smooth_curve(rewards, window=50)
        for threshold in thresholds:
            # Find first episode where smoothed reward >= threshold
            reached = np.where(smoothed >= threshold)[0]
            if len(reached) > 0:
                threshold_data[model].append(reached[0] + 50)  # Account for smoothing window
            else:
                threshold_data[model].append(None)
    
    x = np.arange(len(thresholds))
    width = 0.15
    for idx, model in enumerate(logs.keys()):
        values = [v if v is not None else 0 for v in threshold_data[model]]
        offset = (idx - len(logs) / 2) * width + width / 2
        ax4.bar(x + offset, values, width, label=MODEL_LABELS[model], 
               color=COLORS[model], alpha=0.8)
    
    ax4.set_xlabel('Reward Threshold', fontsize=13)
    ax4.set_ylabel('Episodes to Reach Threshold', fontsize=13)
    ax4.set_title('Training Efficiency - Sample Efficiency', fontsize=15, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{t}' for t in thresholds], fontsize=11)
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('DQN Model Variants - Comprehensive Comparison (10K Episodes)', 
                fontsize=17, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / "comprehensive_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_mha_ablation(logs: dict, output_dir: str):
    """Compare original MHA vs MHA V2."""
    if "mha" not in logs or "mha_v2" not in logs:
        print("⚠ Skipping MHA ablation - missing data")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Rewards comparison
    ax1 = axes[0]
    for model in ["mha", "mha_v2"]:
        rewards = logs[model]["episode_rewards"]
        smoothed = smooth_curve(rewards, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax1.plot(episodes, smoothed, linewidth=3, color=COLORS[model], 
                label=MODEL_LABELS[model], alpha=0.9)
    ax1.set_xlabel("Episode", fontsize=12)
    ax1.set_ylabel("Reward (100-ep Moving Avg)", fontsize=12)
    ax1.set_title("MHA Architecture Comparison - Rewards", fontsize=14, fontweight='bold')
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Loss comparison
    ax2 = axes[1]
    for model in ["mha", "mha_v2"]:
        losses = logs[model]["episode_losses"]
        smoothed = smooth_curve(losses, window=100)
        episodes = range(100, 100 + len(smoothed))
        ax2.plot(episodes, smoothed, linewidth=3, color=COLORS[model], 
                label=MODEL_LABELS[model], alpha=0.9)
    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("Loss (100-ep Moving Avg)", fontsize=12)
    ax2.set_title("MHA Architecture Comparison - Losses", fontsize=14, fontweight='bold')
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('MHA Ablation Study: Original vs Improved Architecture', 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    output_path = Path(output_dir) / "mha_ablation.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def plot_individual_models(logs: dict, output_dir: str):
    """Individual learning curves for each model."""
    n_models = len(logs)
    cols = 3
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(18, 6 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (model, data) in enumerate(logs.items()):
        ax = axes[idx]
        rewards = data["episode_rewards"]
        smoothed = smooth_curve(rewards, window=50)
        
        # Plot raw and smoothed
        ax.plot(range(1, len(rewards) + 1), rewards, alpha=0.15, color=COLORS[model], linewidth=0.5)
        ax.plot(range(50, 50 + len(smoothed)), smoothed, linewidth=2.5, color=COLORS[model])
        
        ax.set_xlabel("Episode", fontsize=11)
        ax.set_ylabel("Reward", fontsize=11)
        ax.set_title(f"{MODEL_LABELS[model]}\nAvg: {data['avg_reward']:.1f}, Best: {data['best_reward']:.0f}", 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Individual Model Learning Curves', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    output_path = Path(output_dir) / "individual_models.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def print_summary(logs: dict):
    """Print summary statistics."""
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"{'Model':<25} {'Episodes':<10} {'Avg Reward':<12} {'Best Reward':<12} {'Improvement':<12}")
    print("-"*80)
    
    base_avg = logs.get("base", {}).get("avg_reward", 0)
    
    for model in logs.keys():
        data = logs[model]
        improvement = ((data["avg_reward"] - base_avg) / base_avg * 100) if base_avg > 0 else 0
        improvement_str = f"+{improvement:.1f}%" if improvement > 0 else f"{improvement:.1f}%"
        print(f"{MODEL_LABELS[model]:<25} {data['total_episodes']:<10} "
              f"{data['avg_reward']:<12.2f} {data['best_reward']:<12.0f} {improvement_str:<12}")
    
    print("="*80 + "\n")


def main():
    """Main function."""
    print("\n" + "="*60)
    print("Generating Comprehensive Comparison Plots")
    print("="*60 + "\n")
    
    os.makedirs(PLOTS_DIR, exist_ok=True)
    
    # Load logs
    logs = load_logs()
    
    if not logs:
        print("❌ No training logs found!")
        return
    
    print("\nGenerating plots...\n")
    
    # Generate plots
    plot_comprehensive_comparison(logs, PLOTS_DIR)
    plot_mha_ablation(logs, PLOTS_DIR)
    plot_individual_models(logs, PLOTS_DIR)
    
    # Print summary
    print_summary(logs)
    
    print(f"✅ All plots saved to: {PLOTS_DIR}/")
    print(f"   - comprehensive_comparison.png")
    print(f"   - mha_ablation.png")
    print(f"   - individual_models.png")


if __name__ == "__main__":
    main()

