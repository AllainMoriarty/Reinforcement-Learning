from typing import Tuple

State = Tuple[int, int]


class GridWorld:
    def __init__(self):
        # 3 rows x 4 columns
        self.rows = 3
        self.cols = 4

        self.start_state: State = (0, 0)
        self.goal_state: State = (0, 3)
        self.bad_state: State = (1, 3)
        self.wall: State = (1, 1)

        self.state = self.start_state

        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

    def reset(self) -> State:
        """Reset the environment to the starting state."""
        self.state = self.start_state
        return self.state

    def step(self, action: str):
        """
        Execute an action.

        Returns:
            next_state
            reward
            done
        """

        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        row, col = self.state

        # 1. Calculate proposed next position
        if action == "UP":
            next_state = (row - 1, col)

        elif action == "DOWN":
            next_state = (row + 1, col)

        elif action == "LEFT":
            next_state = (row, col - 1)

        elif action == "RIGHT":
            next_state = (row, col + 1)

        # 2. Check boundaries
        next_row, next_col = next_state

        outside_grid = (next_row < 0 or next_row >= self.rows or next_col < 0 or next_col >= self.cols)

        if outside_grid:
            next_state = self.state

        # 3. Check wall
        if next_state == self.wall:
            next_state = self.state

        # 4. Move agent
        self.state = next_state

        # 5. Determine reward and termination
        if self.state == self.goal_state:
            reward = 10
            done = True

        elif self.state == self.bad_state:
            reward = -10
            done = True

        else:
            reward = -1
            done = False

        return self.state, reward, done

    def render(self):
        """Print the current GridWorld."""

        for row in range(self.rows):
            cells = []

            for col in range(self.cols):
                position = (row, col)

                if position == self.state:
                    cells.append(" A ")

                elif position == self.goal_state:
                    cells.append("+10")

                elif position == self.bad_state:
                    cells.append("-10")

                elif position == self.wall:
                    cells.append("###")

                else:
                    cells.append(" . ")

            print("|" + "|".join(cells) + "|")

        print()


if __name__ == "__main__":
    env = GridWorld()

    state = env.reset()

    print("Initial state:", state)
    env.render()

    while True:
        action = input("Action [UP/DOWN/LEFT/RIGHT]: ").upper()

        next_state, reward, done = env.step(action)

        print("Next state:", next_state)
        print("Reward:", reward)
        print("Done:", done)

        env.render()

        if done:
            print("Episode finished!")
            break
