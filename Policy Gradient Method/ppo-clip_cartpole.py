import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
  # Build a feedforward neural network
  layers = []
  for j in range(len(sizes) - 1):
    act = activation if j < len(sizes) - 2 else output_activation
    layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
  return nn.Sequential(*layers)

def discount_cumsum(x, discount):
    """
    compute discounted cumulative sums of a vector, e.g. for reward-to-go:
    result[t] = x[t] + discount * x[t+1] + discount^2 * x[t+2] + ...
    """
    # make output array with the same shape as the input
    result = np.zeros_like(x, dtype=np.float32)

    # running total, built up by walking backwards through time
    running_sum = 0.0

    # iterate from the last timestep to the first, so each step can reuse
    # the already-discounted sum of everything that comes after it
    for t in reversed(range(len(x))):
        running_sum = x[t] + discount * running_sum
        result[t] = running_sum
    return result

def compute_gae(rewards, values, gamma=0.99, lam=0.97):
    """
    compute advantage estimates using Generalized Advantage Estimation (GAE-Lambda)
    rewards has length T, values has length T+1 (includes the bootstrap value at the end)
    """
    # make sure inputs are float32 numpy arrays
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    # TD residuals: delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
    # (one-step estimate of how much better the outcome was than the critic expected)
    deltas = (rewards + gamma * values[1:] - values[:-1])

    # GAE-Lambda: discounted sum of TD residuals with discount gamma * lambda
    # lam trades off bias (lam=0, pure one-step TD) vs variance (lam=1, Monte Carlo)
    advantages = discount_cumsum(deltas, gamma * lam)
    return advantages

def train(env_name='CartPole-v1', hidden_sizes=[32], lr=1e-2, epochs=50, batch_size=5000, render=False):
  # make environment, check spaces, get obs / act dims
  render_mode = 'human' if render else None
  env = gym.make(env_name, render_mode=render_mode)
  assert isinstance(env.observation_space, Box), \
    "This example only works for envs with continuous state spaces."
  assert isinstance(env.action_space, Discrete), \
    "This example only works for envs with discrete action spaces."

  obs_dim = env.observation_space.shape[0]
  n_acts = env.action_space.n

  # make core of policy network
  logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

  # make function to compute action distribution
  def get_policy(obs):
    logits = logits_net(obs)
    return Categorical(logits=logits)

  # make action selection function (outputs int actions, sampled from policy)
  def get_action(obs):
    return get_policy(obs).sample().item()

  # make loss function whose gradient, for right data, is policy gradient
  def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

  # make optimizer
  optimizer = Adam(logits_net.parameters(), lr=lr)

  # for training policy
  def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = [] # for observations
    batch_acts = [] # for actions
    batch_weights = [] # for R(tau) weighting in policy gradient
    batch_rets = [] # for measuring episode returns
    batch_lens = [] # for measuring episode lengths

    # reset episode-specific variables
    obs, _ = env.reset() # new API: reset() returns (obs, info)
    done = False # combined terminated/truncated signal
    ep_rews = [] # list for rewards accrued throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
      # rendering
      if (not finished_rendering_this_epoch) and render:
        env.render()

      # save obs
      batch_obs.append(obs.copy())

      # act in the environment
      act = get_action(torch.as_tensor(obs, dtype=torch.float32))
      obs, rew, terminated, truncated, _ = env.step(act)
      done = terminated or truncated

      # save action, reward
      batch_acts.append(act)
      ep_rews.append(rew)

      if done:
        # if episode is over, record info about episode
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_rets.append(ep_ret)
        batch_lens.append(ep_len)

        # the weight for each logprob(a|s) is R(tau)
        batch_weights += [ep_ret] * ep_len

        # reset episode-specific variables
        obs, _ = env.reset()
        done, ep_rews = False, []

        # won't render again this epoch
        finished_rendering_this_epoch = True

        # end experience loop if we have enough of it
        if len(batch_obs) > batch_size:
          break

    # take a single policy gradient update step
    optimizer.zero_grad()
    batch_loss = compute_loss(
        obs=torch.as_tensor(np.array(batch_obs), dtype=torch.float32),
        act=torch.as_tensor(np.array(batch_acts), dtype=torch.int64),
        weights=torch.as_tensor(np.array(batch_weights), dtype=torch.float32)
    )
    batch_loss.backward()
    optimizer.step()
    return batch_loss, batch_rets, batch_lens


  # training loop
  for i in range(epochs):
    batch_loss, batch_rets, batch_lens = train_one_epoch()
    print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))

  env.close()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--lr', type=float, default=1e-2)
  args, _ = parser.parse_known_args()
  print('\nUsing simplest formulation of policy gradient.\n')
  train(env_name=args.env_name, render=args.render, lr=args.lr)
