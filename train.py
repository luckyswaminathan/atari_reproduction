"""
Minimal DQN training loop wiring replay buffer and model.
This is a slim reference implementation; tune hyperparameters before real training.
"""
import ale_py
import argparse
import random
from collections import deque
from typing import Deque, Type
import time
import json
import os
from datetime import datetime

import cv2
import gymnasium as gym
import ale_py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import BaseDQN, DuelingDQN, MHADQN, DuelingMHADQN
from replay_buffer import ReplayBuffer


# Model registry
MODELS = {
    "base": BaseDQN,
    "dueling": DuelingDQN,
    "mha": MHADQN,
    "dueling_mha": DuelingMHADQN,
}

CONFIG = {
    "env_id": "Breakout-v4",
    "buffer_capacity": 50_000,  # Reduced for faster training
    "batch_size": 32,
    "gamma": 0.99,
    "lr": 0.00025,
    "target_update_freq": 1_000,  # Balanced frequency for 2000 episode training
    "warmup_steps": 1_000,  # Reduced warmup for faster start
    "max_episodes": 2000,  # Reduced from 20k to 2k episodes
    "max_steps_per_episode": 5_000,  # Reduced max steps per episode
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 100_000,  # Faster epsilon decay
}


def preprocess(obs: np.ndarray) -> torch.Tensor:
    """Convert RGB observation to normalized 84x84 grayscale tensor shaped (1, 84, 84)."""
    gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(resized).float() / 255.0
    return tensor.unsqueeze(0)


def stack_frames(frames: Deque[torch.Tensor]) -> torch.Tensor:
    """Stack last 4 frames into (4, 84, 84)."""
    return torch.cat(list(frames), dim=0)


def epsilon_by_step(step: int) -> float:
    eps_start, eps_end, eps_decay = (
        CONFIG["epsilon_start"],
        CONFIG["epsilon_end"],
        CONFIG["epsilon_decay"],
    )
    return eps_end + (eps_start - eps_end) * max(0.0, (eps_decay - step) / eps_decay)


def select_action(policy_net: nn.Module, state: torch.Tensor, step: int, num_actions: int) -> int:
    if random.random() < epsilon_by_step(step):
        return random.randrange(num_actions)
    with torch.no_grad():
        q_values = policy_net(state.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())


def train(model_name: str = "base"):
    """
    Train DQN on Atari Breakout.

    Args:
        model_name: Name of the model to use. Options: 'base', 'dueling', 'mha', 'dueling_mha'
    """
    # Track training start time
    training_start_time = time.time()
    training_start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("DEBUG: Starting environment setup...")
    
    # Get model class
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
    
    ModelClass = MODELS[model_name]
    print(f"Selected model: {model_name} ({ModelClass.__name__})")
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir=f"runs/breakout_{model_name}")

    print("DEBUG: Registering ALE environments...")
    gym.register_envs(ale_py)
    print("DEBUG: Creating environment...")
    env = gym.make(CONFIG["env_id"], obs_type="rgb", frameskip=4, render_mode=None)
    print("DEBUG: Environment created successfully!")
    num_actions = env.action_space.n

    policy_net = ModelClass(num_actions).to(device)
    target_net = ModelClass(num_actions).to(device)

    print("\n" + "="*60)
    print("Network Architecture:")
    print("="*60)
    print(policy_net)
    print("="*60 + "\n")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Paper: RMSProp with alpha=0.95, eps=0.01
    optimizer = torch.optim.RMSprop(
        policy_net.parameters(), lr=CONFIG["lr"], alpha=0.95, eps=0.01
    )
    buffer = ReplayBuffer(CONFIG["buffer_capacity"])

    global_step = 0
    episode_rewards = []
    episode_losses = []
    training_started = False

    print(f"Starting training with {CONFIG['max_episodes']} episodes...")
    print(f"Warmup steps: {CONFIG['warmup_steps']}")
    print(f"Target network updates every {CONFIG['target_update_freq']} steps\n")

    for episode in range(CONFIG["max_episodes"]):
        obs, _ = env.reset()
        frames: Deque[torch.Tensor] = deque(maxlen=4)
        first_frame = preprocess(obs)
        for _ in range(4):
            frames.append(first_frame)
        state = stack_frames(frames)

        episode_reward = 0
        episode_loss_sum = 0
        episode_loss_count = 0
        episode_start_time = time.time()

        for step in range(CONFIG["max_steps_per_episode"]):
            state_device = state.to(device)
            action = select_action(policy_net, state_device, global_step, num_actions)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            # Paper: clip rewards to {-1, 0, +1}
            clipped_reward = max(-1.0, min(1.0, reward))
            episode_reward += reward  # Track unclipped for logging

            next_frame = preprocess(next_obs)
            frames.append(next_frame)
            next_state = stack_frames(frames)

            buffer.add(state, action, clipped_reward, next_state, done)
            state = next_state
            global_step += 1

            if len(buffer) >= CONFIG["batch_size"] and global_step > CONFIG["warmup_steps"]:
                if not training_started:
                    training_started = True
                    print("✓ Training started!\n")
                
                states, actions, rewards, next_states, dones = buffer.sample(
                    CONFIG["batch_size"], device
                )

                q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(1)[0]
                    targets = rewards + CONFIG["gamma"] * (1 - dones) * next_q

                loss = F.smooth_l1_loss(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
                optimizer.step()

                episode_loss_sum += loss.item()
                episode_loss_count += 1
                
                # Log to TensorBoard
                writer.add_scalar("Loss/Training", loss.item(), global_step)
                writer.add_scalar("Epsilon", epsilon_by_step(global_step), global_step)
                writer.add_scalar("Buffer/Size", len(buffer), global_step)

                if global_step % CONFIG["target_update_freq"] == 0:
                    target_net.load_state_dict(policy_net.state_dict())
                    writer.add_scalar("Target/Update", 1, global_step)

            if done:
                break

        # Episode summary
        episode_rewards.append(episode_reward)
        avg_loss = episode_loss_sum / episode_loss_count if episode_loss_count > 0 else 0
        episode_losses.append(avg_loss)
        episode_time = time.time() - episode_start_time
        epsilon = epsilon_by_step(global_step)
        
        # Print episode stats every 100 episodes (reduced verbosity)
        if (episode + 1) % 100 == 0:
            print(f"Episode {episode+1}/{CONFIG['max_episodes']} | "
                  f"Reward: {episode_reward:6.1f} | "
                  f"Steps: {step+1:4d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Epsilon: {epsilon:.3f} | "
                  f"Buffer: {len(buffer):6d} | "
                  f"Time: {episode_time:.1f}s")
        
        # Log to TensorBoard
        writer.add_scalar("Reward/Episode", episode_reward, episode)
        writer.add_scalar("Loss/Episode", avg_loss, episode)
        writer.add_scalar("Steps/Episode", step + 1, episode)
        
        # Print running averages every 50 episodes (reduced frequency)
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            avg_loss_recent = np.mean(episode_losses[-50:]) if len(episode_losses) >= 50 else np.mean(episode_losses)
            print(f"\n📊 Last 50 episodes avg: Reward={avg_reward:.1f}, Loss={avg_loss_recent:.4f}\n")
            writer.add_scalar("Reward/Average_50ep", avg_reward, episode)

    env.close()
    writer.close()
    
    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    training_hours = int(total_training_time // 3600)
    training_minutes = int((total_training_time % 3600) // 60)
    training_seconds = int(total_training_time % 60)
    
    # Save trained model
    model_save_path = f"checkpoints/{model_name}_final.pt"
    torch.save({
        'model_name': model_name,
        'model_state_dict': policy_net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'target_net_state_dict': target_net.state_dict(),
        'total_episodes': len(episode_rewards),
        'total_steps': global_step,
        'final_epsilon': epsilon_by_step(global_step),
        'best_reward': max(episode_rewards) if episode_rewards else 0,
        'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0,
        'config': CONFIG,
    }, model_save_path)
    print(f"\n💾 Model saved to: {model_save_path}")
    
    # Save training logs as JSON
    logs_save_path = f"checkpoints/{model_name}_logs.json"
    training_logs = {
        'model_name': model_name,
        'training_start': training_start_datetime,
        'training_end': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total_training_time_seconds': total_training_time,
        'total_training_time_formatted': f"{training_hours}h {training_minutes}m {training_seconds}s",
        'total_episodes': len(episode_rewards),
        'total_steps': global_step,
        'final_epsilon': epsilon_by_step(global_step),
        'config': CONFIG,
        'episode_rewards': episode_rewards,
        'episode_losses': episode_losses,
        'best_reward': max(episode_rewards) if episode_rewards else 0,
        'avg_reward': float(np.mean(episode_rewards)) if episode_rewards else 0,
        'device': str(device),
    }
    with open(logs_save_path, 'w') as f:
        json.dump(training_logs, f, indent=2)
    print(f"📊 Training logs saved to: {logs_save_path}")
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Total steps: {global_step}")
    print(f"Training time: {training_hours}h {training_minutes}m {training_seconds}s")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Final epsilon: {epsilon_by_step(global_step):.3f}")
    print(f"\n📁 Saved files:")
    print(f"   - {model_save_path}")
    print(f"   - {logs_save_path}")
    print(f"\n📈 View training metrics: tensorboard --logdir=runs")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Train DQN on Atari Breakout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  base        - Standard DQN (Mnih et al., 2013)
  dueling     - Dueling DQN architecture
  mha         - DQN with Multi-Head Attention
  dueling_mha - Dueling DQN with Multi-Head Attention

Example:
  python train.py --model base
  python train.py --model dueling
  python train.py -m mha
        """
    )
    
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="base",
        choices=list(MODELS.keys()),
        help=f"Model architecture to use (default: base). Options: {', '.join(MODELS.keys())}"
    )
    
    args = parser.parse_args()
    train(model_name=args.model)


if __name__ == "__main__":
    main()

