import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class MatrixObservationEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = gym.spaces.MultiDiscrete(np.array([1, 2, 3, 4], dtype=np.int64))
        self.action_space = gym.spaces.MultiDiscrete(np.array([3, 4, 3, 4], dtype=np.int64))
        self.state = self.observation_space.sample()

    def reset(self, *, seed=None, options=None):
        del options
        super().reset(seed=seed)
        self.state = self.observation_space.sample()
        return self.state, {}

    def step(self, action):
        assert action.shape == (4,)
        self.state = self.observation_space.sample()
        return self.state, 1.0, False, False, {}


environment = MatrixObservationEnvironment()
model = PPO(
    "MlpPolicy",
    environment,
    n_steps=4,
    batch_size=4,
    n_epochs=1,
    seed=31,
    verbose=1,
    device="cpu",
)
model.learn(total_timesteps=8)
print("training-complete", flush=True)
