import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO


class MatrixActionEnvironment(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        self.observation_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.MultiDiscrete(np.array([2, 2], dtype=np.int64))

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        del options
        super().reset(seed=seed)
        return np.zeros(4, dtype=np.float32), {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        assert action.shape == (2,)
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}


environment = MatrixActionEnvironment()
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
