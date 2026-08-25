import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class StepCallback(BaseCallback):
    def _on_step(self) -> bool:
        print(f"training-step={self.num_timesteps}", flush=True)
        return True


environment = gym.make("CartPole-v1")
model = PPO(
    "MlpPolicy",
    environment,
    n_steps=2,
    batch_size=2,
    n_epochs=1,
    seed=23,
    verbose=1,
    device="cpu",
)
model.learn(total_timesteps=10, callback=StepCallback())
print("training-complete", flush=True)
