from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3 import SAC


class Float64ActionEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    # 初始化与上游 Issue 相同的 float64 连续动作契约
    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float64)
        self._steps = 0

    # 返回确定性的单环境初始 observation
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        self._steps = 0
        return np.zeros(3, dtype=np.float32), {}

    # 完成真实 rollout step，并在首步输出唯一阶段里程碑
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        self._steps += 1
        if self._steps == 1:
            print("training-step=1", flush=True)
        observation = np.full(3, min(self._steps, 10) / 10, dtype=np.float32)
        reward = float(1.0 - np.square(action).mean())
        terminated = self._steps >= 8
        return observation, reward, terminated, False, {}


# 运行足够 rollout 以进入 SAC replay-buffer 梯度更新
def main() -> None:
    env = Float64ActionEnv()
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=16,
        buffer_size=128,
        batch_size=16,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs={"net_arch": [16, 16]},
        seed=0,
        verbose=1,
        device="cpu",
    )
    model.learn(total_timesteps=64, log_interval=1)
    print("training-complete", flush=True)


if __name__ == "__main__":
    main()
