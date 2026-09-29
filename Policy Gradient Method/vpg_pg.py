import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box

# build a simple feed-forward network (multi-layer perceptron)
def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
  layers = []
  for j in range(len(sizes) - 1):
    # hidden layers use `activation`, the last layer uses `output_activation`
    act = activation if j < len(sizes) - 2 else output_activation
    # Linear = weights * input + bias, then apply the activation function
    layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
  # Sequential chains all layers so they run one after another
  return nn.Sequential(*layers)

# discounted cumulative sum: out[t] = x[t] + discount * out[t+1]
# e.g. x=[r0,r1,r2] -> [r0 + g*r1 + g^2*r2, r1 + g*r2, r2]
def discount_cumsum(x, discount):
  result = np.zeros_like(x, dtype=np.float32) # empty array, same length as x
  running_sum = 0 # holds the sum of everything "in the future"
  for t in reversed(range(len(x))): # walk backwards from the last step
    running_sum = x[t] + discount * running_sum # add current value + discounted future
    result[t] = running_sum
  return result

# GAE-Lambda: a smoother, lower-variance estimate of "how much better was this action than average"
def compute_gae(rewards, values, gamma=0.99, lam=0.97):
  # rewards: [r_0 ... r_{T-1}], values: [V(s_0) ... V(s_T)] (one EXTRA value at the end)
  rewards = np.asarray(rewards, dtype=np.float32) # make sure they are numpy float arrays
  values = np.asarray(values, dtype=np.float32)
  # TD residual: delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)  (one-step "surprise")
  deltas = rewards + gamma * values[1:] - values[:-1]
  # advantage = discounted sum of future deltas, discount factor is gamma*lambda
  return discount_cumsum(deltas, gamma * lam)

def train(env_name='CartPole-v1', hidden_sizes=[32], pi_lr=1e-2, vf_lr=1e-2, epochs=50,
          batch_size=5000, gamma=0.99, lam=0.97, train_v_iters=80, render=False):
  # make environment, check spaces, get obs / act dims
  render_mode = 'human' if render else None
  env = gym.make(env_name, render_mode=render_mode)
  assert isinstance(env.observation_space, Box), \
    "This example only works for envs with continuous state spaces."
  assert isinstance(env.action_space, Discrete), \
    "This example only works for envs with discrete action spaces."

  obs_dim = env.observation_space.shape[0] # number of numbers describing a state
  n_acts = env.action_space.n # number of possible actions

  # make policy network: state -> one score (logit) per action
  logits_net = mlp(sizes=[obs_dim] + hidden_sizes + [n_acts])

  # make value network: state -> one number, V(s) = expected future reward
  value_net = mlp(sizes=[obs_dim] + hidden_sizes + [1])

  # make function to compute action distribution
  def get_policy(obs):
    logits = logits_net(obs) # raw scores for each action
    return Categorical(logits=logits) # turn scores into probabilities we can sample from

  # make action selection function (outputs int actions, sampled from policy)
  def get_action(obs):
    return get_policy(obs).sample().item() # sample randomly, .item() -> plain python int

  # make function to estimate V(s) (no gradient needed, only used for data collection)
  def get_value(obs):
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32) # numpy -> torch tensor
    with torch.no_grad(): # don't track gradients, saves memory/time
      value = value_net(obs_tensor)
    return value.item()

  # make policy loss whose gradient is the policy gradient
  # loss = -mean( log pi(a|s) * advantage ), negative because optimizers minimize
  def compute_policy_loss(obs, act, adv):
    logp = get_policy(obs).log_prob(act) # log-probability of the action actually taken
    return -(logp * adv).mean()

  # make value loss: mean squared error between predicted V(s) and the real return
  def compute_value_loss(obs, ret):
    values = value_net(obs).squeeze(-1) # shape (N,1) -> (N,)
    return ((values - ret) ** 2).mean()

  # make optimizers (Adam updates network weights using gradients)
  pi_optimizer = Adam(logits_net.parameters(), lr=pi_lr)
  vf_optimizer = Adam(value_net.parameters(), lr=vf_lr)

  # for training policy
  def train_one_epoch():
    # make some empty lists for logging.
    batch_obs = [] # for observations
    batch_acts = [] # for actions
    batch_advs = [] # for GAE advantages (weights in policy gradient)
    batch_returns = [] # for reward-to-go targets of the value network
    batch_ep_rets = [] # for measuring episode returns
    batch_ep_lens = [] # for measuring episode lengths

    # reset episode-specific variables
    obs, _ = env.reset() # new API: reset() returns (obs, info)
    ep_rews = [] # rewards accrued throughout ep
    ep_vals = [] # value estimates V(s_t) throughout ep

    # render first episode of each epoch
    finished_rendering_this_epoch = False

    # collect experience by acting in the environment with current policy
    while True:
      # rendering
      if (not finished_rendering_this_epoch) and render:
        env.render()

      # save obs
      batch_obs.append(obs.copy()) # copy so later changes don't overwrite it

      # save the critic's value estimate for this state
      ep_vals.append(get_value(obs))

      # act in the environment
      act = get_action(torch.as_tensor(obs, dtype=torch.float32))
      obs, rew, terminated, truncated, _ = env.step(act)
      done = terminated or truncated # episode ended for any reason

      # save action, reward
      batch_acts.append(act)
      ep_rews.append(rew)

      if done:
        # if episode is over, record info about episode
        ep_ret, ep_len = sum(ep_rews), len(ep_rews)
        batch_ep_rets.append(ep_ret)
        batch_ep_lens.append(ep_len)

        # bootstrap value: 0 if truly finished, critic's guess if cut off by time limit
        last_value = 0.0 if terminated else get_value(obs)

        # append it so `values` has one extra element V(s_T)
        values = np.append(ep_vals, last_value)

        # GAE advantage for every step (the weight for each logprob(a|s))
        batch_advs += list(compute_gae(ep_rews, values, gamma, lam))

        # reward-to-go (with bootstrap) = target for the value network
        rews_with_bootstrap = np.append(ep_rews, last_value)
        batch_returns += list(discount_cumsum(rews_with_bootstrap, gamma)[:-1])

        # reset episode-specific variables
        obs, _ = env.reset()
        ep_rews, ep_vals = [], []

        # won't render again this epoch
        finished_rendering_this_epoch = True

        # end experience loop if we have enough of it
        if len(batch_obs) >= batch_size:
          break

    # convert lists to tensors
    obs_t = torch.as_tensor(np.array(batch_obs), dtype=torch.float32)
    act_t = torch.as_tensor(np.array(batch_acts), dtype=torch.int64) # actions are integer ids
    adv_t = torch.as_tensor(np.array(batch_advs), dtype=torch.float32)
    ret_t = torch.as_tensor(np.array(batch_returns), dtype=torch.float32)

    # normalize advantages (mean 0, std 1) -> more stable training; 1e-8 avoids divide-by-zero
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    # take a single policy gradient update step
    pi_optimizer.zero_grad() # clear old gradients
    policy_loss = compute_policy_loss(obs_t, act_t, adv_t)
    policy_loss.backward() # compute gradients
    pi_optimizer.step() # update policy weights

    # fit the value function with several gradient steps on the same batch
    for _ in range(train_v_iters):
      vf_optimizer.zero_grad()
      value_loss = compute_value_loss(obs_t, ret_t)
      value_loss.backward()
      vf_optimizer.step()

    return policy_loss.item(), value_loss.item(), batch_ep_rets, batch_ep_lens

  # training loop
  for i in range(epochs):
    pi_loss, v_loss, batch_rets, batch_lens = train_one_epoch()
    print('epoch: %3d \t pi_loss: %.3f \t v_loss: %.3f \t return: %.3f \t ep_len: %.3f' %
              (i, pi_loss, v_loss, np.mean(batch_rets), np.mean(batch_lens)))

  env.close()

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '--env', type=str, default='CartPole-v1')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--pi_lr', type=float, default=1e-2)
  parser.add_argument('--vf_lr', type=float, default=1e-2)
  args, _ = parser.parse_known_args()
  print('\nVanilla Policy Gradient + Value Function + GAE-Lambda\n')
  train(env_name=args.env_name, render=args.render, pi_lr=args.pi_lr, vf_lr=args.vf_lr)
