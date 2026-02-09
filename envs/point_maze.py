import numpy as np
import gym
from gym import spaces
from typing import Tuple, Dict, Optional


class PointMaze2D(gym.Env):
    """
    2D Point Robot Maze Navigation Environment

    State space: [x, y] position coordinates
    Action space: [dx, dy] movement vector
    Goal: Navigate from start to goal position

    Maze layout:
    +-----------+
    |  S        |
    |   ####    |
    |   #  #    |
    |   ####    |
    |        G  |
    +-----------+

    S: Start region
    G: Goal region
    #: Obstacle
    """

    def __init__(
        self,
        maze_size: float = 10.0,
        goal_radius: float = 0.5,
        max_steps: int = 200,
        action_scale: float = 0.5,
        sparse_reward: bool = True,
    ):
        super().__init__()

        self.maze_size = maze_size
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.sparse_reward = sparse_reward

        # State and action space
        self.observation_space = spaces.Box(
            low=-maze_size, high=maze_size, shape=(2,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Start and goal positions
        self.start_pos = np.array([-4.0, 4.0])
        self.goal_pos = np.array([4.0, -4.0])

        # Obstacle definitions (rectangle: [x_min, x_max, y_min, y_max])
        self.obstacles = [
            [-2.0, 2.0, -2.0, 2.0],  # Central obstacle
        ]

        # State
        self.pos = None
        self.step_count = 0

    def reset(self, seed=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment"""
        if seed is not None:
            np.random.seed(seed)

        # Random initialization near start position
        self.pos = self.start_pos + np.random.uniform(-0.5, 0.5, 2)
        self.step_count = 0

        return self.pos.copy().astype(np.float32), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action"""
        action = np.clip(action, -1.0, 1.0)

        # Compute new position
        new_pos = self.pos + action * self.action_scale

        # Boundary check
        new_pos = np.clip(new_pos, -self.maze_size, self.maze_size)

        # Obstacle collision check
        if not self._check_collision(new_pos):
            self.pos = new_pos

        self.step_count += 1

        # Compute reward
        dist_to_goal = np.linalg.norm(self.pos - self.goal_pos)
        reached_goal = dist_to_goal < self.goal_radius

        if self.sparse_reward:
            reward = 1.0 if reached_goal else 0.0
        else:
            reward = -dist_to_goal / 10.0 + (1.0 if reached_goal else 0.0)

        # Termination condition
        terminated = reached_goal
        truncated = self.step_count >= self.max_steps

        info = {
            "dist_to_goal": dist_to_goal,
            "success": reached_goal,
        }

        return self.pos.copy().astype(np.float32), reward, terminated, truncated, info

    def _check_collision(self, pos: np.ndarray) -> bool:
        """Check collision with obstacles"""
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            if x_min <= pos[0] <= x_max and y_min <= pos[1] <= y_max:
                return True
        return False

    def render(self, mode="rgb_array"):
        """Render environment (return image array)"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle, Circle

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        # Draw boundary
        ax.set_xlim(-self.maze_size, self.maze_size)
        ax.set_ylim(-self.maze_size, self.maze_size)

        # Draw obstacles
        for obs in self.obstacles:
            x_min, x_max, y_min, y_max = obs
            rect = Rectangle(
                (x_min, y_min),
                x_max - x_min,
                y_max - y_min,
                facecolor="gray",
                edgecolor="black",
            )
            ax.add_patch(rect)

        # Draw goal
        goal_circle = Circle(
            self.goal_pos, self.goal_radius, facecolor="green", alpha=0.5, label="Goal"
        )
        ax.add_patch(goal_circle)

        # Draw current position
        if self.pos is not None:
            ax.plot(self.pos[0], self.pos[1], "bo", markersize=10, label="Agent")

        # Draw start
        ax.plot(
            self.start_pos[0], self.start_pos[1], "r^", markersize=10, label="Start"
        )

        ax.set_aspect("equal")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Convert to array
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return image


def generate_offline_dataset(
    env: PointMaze2D, num_trajectories: int = 100, policy_type: str = "mixed"
) -> Dict[str, np.ndarray]:
    """
    Generate offline dataset

    Args:
        env: Environment instance
        num_trajectories: Number of trajectories
        policy_type: Policy type
            - 'random': Random policy
            - 'expert': Expert policy (move toward goal)
            - 'mixed': Mixed policy

    Returns:
        dataset: Dictionary containing observations, actions, rewards, next_observations, terminals
    """
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    for traj_idx in range(num_trajectories):
        obs, _ = env.reset()
        done = False

        while not done:
            # Select action
            if policy_type == "random":
                action = env.action_space.sample()
            elif policy_type == "expert":
                # Move toward goal with slight noise
                direction = env.goal_pos - obs
                direction = direction / (np.linalg.norm(direction) + 1e-8)
                action = direction + np.random.randn(2) * 0.1
                action = np.clip(action, -1, 1)
            elif policy_type == "mixed":
                # 50% random, 50% expert
                if np.random.rand() < 0.5:
                    action = env.action_space.sample()
                else:
                    direction = env.goal_pos - obs
                    direction = direction / (np.linalg.norm(direction) + 1e-8)
                    action = direction + np.random.randn(2) * 0.2
                    action = np.clip(action, -1, 1)
            else:
                action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            terminals.append(terminated)

            obs = next_obs

        if (traj_idx + 1) % 20 == 0:
            print(f"Generated trajectory {traj_idx + 1}/{num_trajectories}")

    return {
        "observations": np.array(observations, dtype=np.float32),
        "actions": np.array(actions, dtype=np.float32),
        "rewards": np.array(rewards, dtype=np.float32),
        "next_observations": np.array(next_observations, dtype=np.float32),
        "terminals": np.array(terminals, dtype=np.float32),
    }


def visualize_dataset(
    dataset: Dict[str, np.ndarray], env: PointMaze2D, save_path: Optional[str] = None
):
    """Visualize dataset distribution"""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # State distribution
    ax = axes[0]
    ax.scatter(
        dataset["observations"][:, 0],
        dataset["observations"][:, 1],
        c=dataset["rewards"],
        cmap="viridis",
        alpha=0.3,
        s=1,
    )

    # Draw obstacles
    for obs in env.obstacles:
        x_min, x_max, y_min, y_max = obs
        rect = Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            facecolor="gray",
            edgecolor="black",
            alpha=0.7,
        )
        ax.add_patch(rect)

    # Draw goal
    goal_circle = Circle(env.goal_pos, env.goal_radius, facecolor="green", alpha=0.5)
    ax.add_patch(goal_circle)

    ax.set_xlim(-env.maze_size, env.maze_size)
    ax.set_ylim(-env.maze_size, env.maze_size)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("State Space Distribution")
    ax.set_aspect("equal")

    # Action distribution
    ax = axes[1]
    ax.scatter(dataset["actions"][:, 0], dataset["actions"][:, 1], alpha=0.1, s=1)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("dx")
    ax.set_ylabel("dy")
    ax.set_title("Action Space Distribution")
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Test environment
    env = PointMaze2D()

    obs, _ = env.reset()
    print(f"Initial state: {obs}")

    # Execute a few steps
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(
            f"Step {i + 1}: state={obs}, reward={reward}, dist={info['dist_to_goal']:.2f}"
        )

    # Generate dataset
    print("\nGenerating offline dataset...")
    dataset = generate_offline_dataset(env, num_trajectories=20, policy_type="mixed")
    print(f"Dataset size: {len(dataset['observations'])}")

    # Visualize
    try:
        visualize_dataset(dataset, env, save_path="./point_maze_dataset.png")
    except Exception as e:
        print(f"Visualization skipped: {e}")

    print("2D Point environment test passed!")
