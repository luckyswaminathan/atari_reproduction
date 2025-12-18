#!/usr/bin/env python3
"""
Extract training data from TensorBoard events file to JSON format.
Matches the structure of logs saved by train.py
"""
import json
import os
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def extract_tensorboard_to_json(events_file_path: str, output_json_path: str):
    """
    Extract episode rewards and losses from TensorBoard events file.
    
    Args:
        events_file_path: Path to TensorBoard events file (.tfevents.*)
        output_json_path: Path to save JSON log file
    """
    print(f"Reading TensorBoard events from: {events_file_path}")
    
    # Load TensorBoard events
    ea = EventAccumulator(str(Path(events_file_path).parent))
    ea.Reload()
    
    # Extract scalars
    episode_rewards = []
    episode_losses = []
    
    # Get Reward/Episode scalar
    if 'Reward/Episode' in ea.Tags()['scalars']:
        reward_events = ea.Scalars('Reward/Episode')
        episode_rewards = [event.value for event in reward_events]
        print(f"Found {len(episode_rewards)} reward entries")
    else:
        print("Warning: 'Reward/Episode' not found in TensorBoard logs")
    
    # Get Loss/Episode scalar
    if 'Loss/Episode' in ea.Tags()['scalars']:
        loss_events = ea.Scalars('Loss/Episode')
        episode_losses = [event.value for event in loss_events]
        print(f"Found {len(episode_losses)} loss entries")
    else:
        print("Warning: 'Loss/Episode' not found in TensorBoard logs")
    
    # Get total steps from last episode
    total_steps = 0
    if 'Steps/Episode' in ea.Tags()['scalars']:
        step_events = ea.Scalars('Steps/Episode')
        if step_events:
            # Sum all steps
            total_steps = sum(event.value for event in step_events)
    
    # Calculate statistics
    total_episodes = len(episode_rewards)
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    best_reward = max(episode_rewards) if episode_rewards else 0
    
    # Create JSON structure matching train.py output
    log_data = {
        "model_name": "mha_v2",
        "training_start": "2025-12-15 04:24:00",  # Approximate start time
        "training_end": "2025-12-15 14:24:00",  # Approximate end time
        "total_training_time_seconds": 36000.0,  # Approximate
        "total_training_time_formatted": "10h 0m 0s",
        "total_episodes": total_episodes,
        "total_steps": int(total_steps),
        "final_epsilon": 0.1,
        "config": {
            "env_id": "Breakout-v4",
            "buffer_capacity": 50000,
            "batch_size": 32,
            "gamma": 0.99,
            "lr": 0.00025,
            "target_update_freq": 1000,
            "warmup_steps": 1000,
            "max_episodes": 50000,
            "max_steps_per_episode": 5000,
            "epsilon_start": 1.0,
            "epsilon_end": 0.1,
            "epsilon_decay": 100000
        },
        "episode_rewards": episode_rewards,
        "episode_losses": episode_losses,
        "best_reward": best_reward,
        "avg_reward": avg_reward,
        "device": "cuda"
    }
    
    # Save to JSON
    os.makedirs(Path(output_json_path).parent, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    print(f"\n✅ Extracted {total_episodes} episodes")
    print(f"   Avg reward: {avg_reward:.2f}")
    print(f"   Best reward: {best_reward:.0f}")
    print(f"   Saved to: {output_json_path}")


if __name__ == "__main__":
    events_file = "checkpoint_20k_mhaV2/breakout_mha_v2/events.out.tfevents.1765772658.ip-172-31-41-240.ec2.internal.7129.0"
    output_file = "checkpoint_20k_mhaV2/mha_v2_logs.json"
    
    extract_tensorboard_to_json(events_file, output_file)

