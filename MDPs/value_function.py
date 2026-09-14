from collections import defaultdict

from gridworld import GridWorld
from trajectory import generate_trajectory


def compute_returns(trajectory, gamma=0.9):
    """
    Compute discounted return G_t for every step in one trajectory.

    G_t = r_{t+1} + gamma*r_{t+2} + gamma^2*r_{t+3} + ...
    """

    returns = [0.0] * len(trajectory)

    G = 0.0

    # Work backwards through the trajectory
    for t in reversed(range(len(trajectory))):
        reward = trajectory[t]["reward"]

        G = reward + gamma * G

        returns[t] = G

    return returns


def estimate_value_function(env, num_episodes=10000, gamma=0.9):
    """
    Estimate V^pi(s) using Monte Carlo sampling.

    V^pi(s) ≈ average return observed after visiting state s.
    """

    returns_sum = defaultdict(float)
    visit_count = defaultdict(int)

    for episode in range(num_episodes):
        trajectory = generate_trajectory(env)

        returns = compute_returns(
            trajectory,
            gamma=gamma,
        )

        visited_states = set()

        for t, transition in enumerate(trajectory):
            state = transition["state"]

            # First-visit Monte Carlo:
            # only use the first occurrence of a state
            # in this episode.
            if state in visited_states:
                continue

            visited_states.add(state)

            returns_sum[state] += returns[t]
            visit_count[state] += 1

    value_function = {}

    for state in returns_sum:
        value_function[state] = (
            returns_sum[state]
            / visit_count[state]
        )

    return value_function


def print_value_function(env, value_function):
    print("\nEstimated V(s):\n")

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
                value = value_function.get(state, 0.0)
                cells.append(f"{value:6.2f}")

        print("|" + "|".join(cells) + "|")


if __name__ == "__main__":
    env = GridWorld()

    values = estimate_value_function(
        env,
        num_episodes=10000,
        gamma=0.9,
    )

    print_value_function(env, values)