import numpy as np
import torch
from agent import Agent
from torch import Tensor, nn, optim

from sensor_array import SensorArray


class ActorCriticModel(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_inputs, 128),
            nn.ReLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_outputs),
            nn.Tanh(),  # Steering and acceleration are bounded [-1, 1]
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)  # State-value output
        )

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    # def save(self, dir_path="./checkpoints", tag="latest"):
    #     os.makedirs(dir_path, exist_ok=True)
    #     torch.save(
    #         {
    #             "actor_state_dict": self.actor.state_dict(),
    #             "critic_state_dict": self.critic.state_dict(),
    #             "actor_optimizer_state_dict": self.actor_optim.state_dict(),
    #             "critic_optimizer_state_dict": self.critic_optim.state_dict(),
    #         },
    #         os.path.join(dir_path, f"agent_{tag}.pt"),
    #     )
    #     carlos_logging.log_message(
    #         f"Saved model checkpoint to {dir_path}/agent_{tag}.pt"
    #     )

    # def load(self, path):
    #     checkpoint = torch.load(path)
    #     self.actor.load_state_dict(checkpoint["actor_state_dict"])
    #     self.critic.load_state_dict(checkpoint["critic_state_dict"])
    #     self.actor_optim.load_state_dict(checkpoint["actor_optimizer_state_dict"])
    #     self.critic_optim.load_state_dict(checkpoint["critic_optimizer_state_dict"])
    #     carlos_logging.log_message(f"Loaded model checkpoint from {path}")


class NewAgent(Agent):
    def __init__(
        self,
        sensor_array: SensorArray,
        lr=1e-3,
        input_scaler=None,
        max_accel: float = 10.0,
    ):
        self.num_inputs = 2 + sensor_array.num_sensors  # speed,
        self.input_scaler = input_scaler or torch.tensor(
            [0.001] * self.num_inputs + [np.pi, 70.0]
        )
        self.max_accel = max_accel
        self.model = ActorCriticModel(self.num_inputs, 2)
        self.optim = optim.Adam(self.model.parameters(), lr=lr)

        super().__init__(sensor_array)

    def decide(self, state: Tensor) -> Tensor:
        # Warning: doesn't handle multiple decisions. Only provide a single dimensional array
        action, _critic_out = self.model(state.reshape(-1, self.num_inputs))
        self.last_state = state

        return (action * torch.tensor([np.pi, self.max_accel])).flatten()

    def compute_reward(self, state, in_lane, in_motion):
        speed, steering_angle, *sensor_data = state
        reward = 0.0

        # 1. Stay in lane
        reward += 1.0 if in_lane else -2.0

        # 2. Encourage motion
        reward += 0.5 if in_motion else -0.5

        # 3. Encourage higher speed (normalized)
        reward += 0.1 * (speed / 75.0)

        # 4. Penalize proximity to obstacles
        min_sensor = min(sensor_data)
        if min_sensor < 1.0:
            reward -= 1.0 - min_sensor

        # 6. Small time penalty
        reward -= 0.01

        return torch.tensor([reward])

    def train_step(self, next_state: Tensor, reward: Tensor, done: bool):
        # Convert next_state to proper shape
        next_state = next_state.reshape(-1, self.num_inputs)

        # Get value estimate and action from previous state
        with torch.no_grad():
            _, next_value = self.model(next_state)
            target = reward + (
                0.99 * next_value * (1.0 - float(done))
            )  # discounted future reward

        # Get current action and value estimate
        action, value = self.model(self.last_state.reshape(-1, self.num_inputs))

        # Critic loss (MSE)
        value_loss = nn.functional.mse_loss(value, target)

        # Actor loss (encourage actions that lead to higher value)
        log_prob = -(
            (action - action.detach()) ** 2
        ).mean()  # placeholder for continuous action log_prob
        advantage = (target - value).detach()
        actor_loss = -log_prob * advantage

        loss = value_loss + actor_loss

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # return loss.item()
