import random
from gridworld import GridWorld


def random_policy(state, actions):
    """
    A policy decides which action to take in a state.

    This policy is intentionally stupid:
    it ignores the state and chooses a random action.
    """
    return random.choice(actions)


def generate_trajectory(env, max_steps=50):
    """
    Run one episode in the environment.

    Returns a trajectory containing:
    (state, action, reward, next_state)
    """

    trajectory = []

    state = env.reset()

    for step in range(max_steps):
        action = random_policy(state, env.actions)

        next_state, reward, done = env.step(action)

        transition = {
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
        }

        trajectory.append(transition)

        state = next_state

        if done:
            break

    return trajectory


def print_trajectory(trajectory):
    print("\nTrajectory:\n")

    for t, transition in enumerate(trajectory):
        print(
            f"t={t}: "
            f"{transition['state']} "
            f"--{transition['action']}--> "
            f"{transition['next_state']} "
            f"reward={transition['reward']}"
        )


def total_reward(trajectory):
    return sum(
        transition["reward"]
        for transition in trajectory
    )


if __name__ == "__main__":
    env = GridWorld()

    trajectory = generate_trajectory(env)

    print_trajectory(trajectory)

    print("\nTotal reward:", total_reward(trajectory))