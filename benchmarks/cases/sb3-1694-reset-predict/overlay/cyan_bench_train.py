from __future__ import annotations

import numpy as np
from gymnasium import Env, spaces
from stable_baselines3 import PPO


class ContractEnv(Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._step = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        del options
        self._step = 0
        return np.zeros(5, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        del action
        self._step += 1
        observation = np.full(5, self._step / 32, dtype=np.float32)
        return observation, 1.0, self._step >= 32, False, {}


model = PPO(
    "MlpPolicy",
    ContractEnv(),
    n_steps=32,
    batch_size=32,
    n_epochs=1,
    policy_kwargs={"net_arch": [8]},
    verbose=1,
    seed=13,
    device="cpu",
)
model.learn(total_timesteps=128)
print("training-main-loop-complete", flush=True)
evaluation = ContractEnv()
observation, _ = evaluation.reset()
model.predict(observation, deterministic=True)
print("post-training-predict-complete", flush=True)
