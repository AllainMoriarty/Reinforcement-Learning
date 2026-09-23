from MDPs.gridworld import GridWorld

def get_next_state(env, state, action):
    """
    Predict the next state without changing env.state.
    This reproduces the transition rules from GridWorld.step().
    """
    row, col = state

    if action == "UP":
        next_state = (row - 1, col)
    elif action == "DOWN":
        next_state = (row + 1, col)
    elif action == "LEFT":
        next_state = (row, col - 1)
    elif action == "RIGHT":
        next_state = (row, col + 1)
    else:
        raise ValueError(f"Invalid action: {action}")

    next_row, next_col = next_state
    outside_grid = (next_row < 0 or next_row >= env.rows or next_col < 0 or next_col >= env.cols)

    # If action hits boundary or wall, stay in same state
    if outside_grid or next_state == env.wall:
        return state

    return next_state

def get_reward(env, next_state):
  """
  Reward received after entering next_state
  """
  if next_state == env.goal_state:
    return 10

  if next_state == env.bad_state:
    return -10

  return -1

def get_states(env):
  """
  Return all valid states except the wall
  """
  states = []
  for row in range(env.rows):
    for col in range(env.cols):
      state = (row, col)
      if state != env.wall:
        states.append(state)

  return states

def policy_evaluation(env, policy, values, gamma=0.9, theta=1e-6):
    """
    Evaluate the current policy.
    Bellman expectation equation:

        V^pi(s) = R(s, pi(s)) + gamma * V^pi(s')

    Because our policy and environment are deterministic,
    there is only one action and one next state.
    """
    while True:
      delta = 0
      new_values = values.copy()

      for state in values:
        # Terminal states have no future value
        if state == env.goal_state or state == env.bad_state:
          continue

        # Follow the current policy
        action = policy[state]
        next_state = get_next_state(env, state, action)
        reward = get_reward(env, next_state)
        new_value = (reward + gamma * values[next_state])
        new_values[state] = new_value
        delta = max(delta, abs(new_value - values[state]))

      values = new_values

      # V^pi has converged
      if delta < theta:
        break

    return values

def policy_improvement(env, policy, values, gamma=0.9):
    """
    Improve the policy greedily using V^pi.
    pi_new(s) = argmax_a [R(s,a) + gamma * V^pi(s')]
    """
    policy_stable = True

    for state in values:
      if state == env.goal_state or state == env.bad_state:
        continue
      old_action = policy[state]
      best_action = None
      best_value = float("-inf")

      # Try every possible action
      for action in env.actions:
        next_state = get_next_state(env, state, action)
        reward = get_reward(env, next_state)
        action_value = (reward + gamma * values[next_state])
        if action_value > best_value:
          best_value = action_value
          best_action = action

      # Replace old policy with better action
      policy[state] = best_action
      if best_action != old_action:
        policy_stable = False

    return policy, policy_stable

def policy_iteration(env, gamma=0.9, theta=1e-6):
    """
    Policy Iteration:
        1. Initialize arbitrary policy
        2. Policy Evaluation
        3. Policy Improvement
        4. Repeat until policy stops changing
    """
    states = get_states(env)

    # V(s) = 0 initially
    values = {state: 0.0 for state in states}

    # Arbitrary initial policy, start by telling the agent to always go UP
    policy = {}
    for state in states:
      if state == env.goal_state or state == env.bad_state:
        continue
      policy[state] = "UP"

    iteration = 0
    while True:
      iteration += 1
      print(f"\n=== Policy Iteration {iteration} ===")

      # 1. Policy Evaluation
      values = policy_evaluation(env, policy, values, gamma, theta)
      print("\nAfter Policy Evaluation:")
      print_values(env, values)

      # 2.  Policy Improvement
      policy, policy_stable = policy_improvement(env, policy, values, gamma=gamma)
      print("\nAfter Policy Improvement:")
      print_policy(env, policy)

      # If policy no longer changes, we have reached the optimal policy
      if policy_stable:
        print("\nPolicy is stable!")
        break

    return values, policy

def print_values(env, values):
    print("\nOptimal Value Function V*(s):\n")

    for row in range(env.rows):
        cells = []

        for col in range(env.cols):
            state = (row, col)
            if state == env.wall:
                cells.append(" WALL ")
            elif state == env.goal_state:
                cells.append(" GOAL ")
            elif state == env.bad_state:
                cells.append(" BAD  ")
            else:
                cells.append(f"{values[state]:6.2f}")

        print("|" + "|".join(cells) + "|")


def print_policy(env, policy):
    symbols = {
        "UP": " ↑ ",
        "DOWN": " ↓ ",
        "LEFT": " ← ",
        "RIGHT": " → ",
    }

    print("\nOptimal Policy π*(s):\n")

    for row in range(env.rows):
        cells = []

        for col in range(env.cols):
            state = (row, col)
            if state == env.wall:
                cells.append("###")
            elif state == env.goal_state:
                cells.append(" G ")
            elif state == env.bad_state:
                cells.append(" X ")
            else:
                action = policy[state]
                cells.append(symbols[action])

        print("|" + "|".join(cells) + "|")

if __name__ == "__main__":
    env = GridWorld()
    values, policy = policy_iteration(env, gamma=0.9)

    print("\n======================")
    print("Final Value Function")
    print("======================")
    print_values(env, values)

    print("\n======================")
    print("Final Optimal Policy")
    print("======================")
    print_policy(env, policy)