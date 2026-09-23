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


def value_iteration(env, gamma=0.9, theta=1e-6):
    """
    Find the optimal value function V*(s) using Value Iteration.
    Bellman optimality equation:

        V*(s) = max_a [R(s, a) + gamma * V*(s')]

    Because our GridWorld is deterministic, each action leads to exactly one next state.
    """

    values = {}

    # Initialize all valid states with value 0
    for row in range(env.rows):
        for col in range(env.cols):
            state = (row, col)
            if state == env.wall:
                continue
            values[state] = 0.0

    while True:
        delta = 0
        new_values = values.copy()

        for state in values:
            # Terminal states have no future value
            if state == env.goal_state or state == env.bad_state:
                continue

            action_values = []
            for action in env.actions:
                next_state = get_next_state(env, state, action)
                reward = get_reward(env, next_state)
                value = (reward + gamma * values[next_state])
                action_values.append(value)

            # Bellman optimality update
            best_value = max(action_values)
            new_values[state] = best_value
            delta = max(delta, abs(best_value - values[state]))

        values = new_values

        # Stop when values stop changing significantly
        if delta < theta:
            break

    return values


def extract_policy(env, values, gamma=0.9):
    """
    After finding V*(s), extract the best action in every state.
    pi*(s) = argmax_a [R(s,a) + gamma * V*(s')]
    """
    policy = {}
    for state in values:
        if state == env.goal_state:
            continue

        if state == env.bad_state:
            continue

        best_action = None
        best_value = float("-inf")

        for action in env.actions:
            next_state = get_next_state(env, state, action)
            reward = get_reward(env, next_state)
            action_value = (reward + gamma * values[next_state])

            if action_value > best_value:
                best_value = action_value
                best_action = action

        policy[state] = best_action

    return policy


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
    values = value_iteration(env, gamma=0.9)
    policy = extract_policy(env, values, gamma=0.9)

    print_values(env, values)
    print_policy(env, policy)