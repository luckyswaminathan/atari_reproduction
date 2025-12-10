"""
Minimal DQN training loop wiring replay buffer and model.
This is a slim reference implementation; tune hyperparameters before real training.
"""
import ale_py
import random
from collections import deque
from typing import Deque
import time

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model import DQN
from replay_buffer import ReplayBuffer


CONFIG = {
    "env_id": "ALE/Breakout-v5",
    "buffer_capacity": 100_000,
    "batch_size": 32,
    "gamma": 0.99,
    "lr": 1e-4,
    "target_update_freq": 1_000,
    "warmup_steps": 5_000,
    "max_episodes": 20000,  # keep small by default
    "max_steps_per_episode": 10_000,
    "epsilon_start": 1.0,
    "epsilon_end": 0.1,
    "epsilon_decay": 100_000,  # steps
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


def select_action(policy_net: DQN, state: torch.Tensor, step: int, num_actions: int) -> int:
    if random.random() < epsilon_by_step(step):
        return random.randrange(num_actions)
    with torch.no_grad():
        q_values = policy_net(state.unsqueeze(0))
        return int(torch.argmax(q_values, dim=1).item())


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # TensorBoard logging
    writer = SummaryWriter(log_dir="runs/breakout_dqn")
    
    env = gym.make(CONFIG["env_id"], obs_type="rgb", frameskip=4, render_mode=None)
    num_actions = env.action_space.n

    policy_net = DQN(num_actions).to(device)
    target_net = DQN(num_actions).to(device)

    print("\n" + "="*60)
    print("Network Architecture:")
    print("="*60)
    print(policy_net)
    print("="*60 + "\n")
    
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=CONFIG["lr"])
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
            episode_reward += reward

            next_frame = preprocess(next_obs)
            frames.append(next_frame)
            next_state = stack_frames(frames)

            buffer.add(state, action, reward, next_state, done)
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
        
        # Print episode stats
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
        
        # Print running averages every 10 episodes
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_loss_recent = np.mean(episode_losses[-10:])
            print(f"\n📊 Last 10 episodes avg: Reward={avg_reward:.1f}, Loss={avg_loss_recent:.4f}\n")
            writer.add_scalar("Reward/Average_10ep", avg_reward, episode)

    env.close()
    writer.close()
    
    # Final summary
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total episodes: {len(episode_rewards)}")
    print(f"Total steps: {global_step}")
    print(f"Average reward: {np.mean(episode_rewards):.2f}")
    print(f"Best episode reward: {max(episode_rewards):.2f}")
    print(f"Final epsilon: {epsilon_by_step(global_step):.3f}")
    print(f"\n📈 View training metrics: tensorboard --logdir=runs")
    print("="*60)


if __name__ == "__main__":
    train()

