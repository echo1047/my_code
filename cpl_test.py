import argparse
import random
import time
import os

import gym
import d4rl

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
from torch.nn import functional as F
from typing import Dict, List, Union, Tuple, Optional
from tqdm import tqdm

# from offlinerlkit.nets import MLP
# from offlinerlkit.modules import ActorProb, Critic, TanhDiagGaussian
from offline_rl.modules.actor_module import ActorProb
from offline_rl.modules.critic_module import Critic
from offline_rl.modules.dist_module import TanhDiagGaussian
# from offlinerlkit.buffer import ReplayBuffer
# from offlinerlkit.utils.logger import Logger, make_log_dirs
# from offlinerlkit.policy_trainer import MFPolicyTrainer
# from offlinerlkit.policy import CQLPolicy


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="cql")
    parser.add_argument("--task", type=str, default="maze2d-umaze-v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dims", type=int, nargs='*', default=[256, 256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--target-entropy", type=int, default=None)
    parser.add_argument("--auto-alpha", default=False) #! True
    parser.add_argument("--alpha-lr", type=float, default=1e-4)
    parser.add_argument("--cql-weight", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-q-backup", type=bool, default=False)
    parser.add_argument("--deterministic-backup", type=bool, default=True)
    parser.add_argument("--with-lagrange", type=bool, default=False)
    parser.add_argument("--lagrange-threshold", type=float, default=10.0)
    parser.add_argument("--cql-alpha-lr", type=float, default=3e-4)
    parser.add_argument("--num-repeat-actions", type=int, default=10)
    
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    return parser.parse_args()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        output_dim: Optional[int] = None,
        activation: nn.Module = nn.ReLU,
        dropout_rate: Optional[float] = None
    ) -> None:
        super().__init__()
        hidden_dims = [input_dim] + list(hidden_dims)
        model = []
        for in_dim, out_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            model += [nn.Linear(in_dim, out_dim), activation()]
            if dropout_rate is not None:
                model += [nn.Dropout(p=dropout_rate)]

        self.output_dim = hidden_dims[-1]
        if output_dim is not None:
            model += [nn.Linear(hidden_dims[-1], output_dim)]
            self.output_dim = output_dim
        self.model = nn.Sequential(*model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    
class ReplayBuffer:
    def __init__(
        self,
        buffer_size: int,
        obs_shape: Tuple,
        obs_dtype: np.dtype,
        action_dim: int,
        action_dtype: np.dtype,
        device: str = "cpu"
    ) -> None:
        self._max_size = buffer_size
        self.obs_shape = obs_shape
        self.obs_dtype = obs_dtype
        self.action_dim = action_dim
        self.action_dtype = action_dtype

        self._ptr = 0
        self._size = 0

        self.observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.next_observations = np.zeros((self._max_size,) + self.obs_shape, dtype=obs_dtype)
        self.actions = np.zeros((self._max_size, self.action_dim), dtype=action_dtype)
        self.rewards = np.zeros((self._max_size, 1), dtype=np.float32)
        self.terminals = np.zeros((self._max_size, 1), dtype=np.float32)

        self.device = torch.device(device)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        terminal: np.ndarray
    ) -> None:
        # Copy to avoid modification by reference
        self.observations[self._ptr] = np.array(obs).copy()
        self.next_observations[self._ptr] = np.array(next_obs).copy()
        self.actions[self._ptr] = np.array(action).copy()
        self.rewards[self._ptr] = np.array(reward).copy()
        self.terminals[self._ptr] = np.array(terminal).copy()

        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)
    
    def add_batch(
        self,
        obss: np.ndarray,
        next_obss: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        terminals: np.ndarray
    ) -> None:
        batch_size = len(obss)
        indexes = np.arange(self._ptr, self._ptr + batch_size) % self._max_size

        self.observations[indexes] = np.array(obss).copy()
        self.next_observations[indexes] = np.array(next_obss).copy()
        self.actions[indexes] = np.array(actions).copy()
        self.rewards[indexes] = np.array(rewards).copy()
        self.terminals[indexes] = np.array(terminals).copy()

        self._ptr = (self._ptr + batch_size) % self._max_size
        self._size = min(self._size + batch_size, self._max_size)
    
    def load_dataset(self, dataset: Dict[str, np.ndarray]) -> None:
        observations = np.array(dataset["observations"], dtype=self.obs_dtype)
        next_observations = np.array(dataset["next_observations"], dtype=self.obs_dtype)
        actions = np.array(dataset["actions"], dtype=self.action_dtype)
        rewards = np.array(dataset["rewards"], dtype=np.float32).reshape(-1, 1)
        terminals = np.array(dataset["terminals"], dtype=np.float32).reshape(-1, 1)

        self.observations = observations
        self.next_observations = next_observations
        self.actions = actions
        self.rewards = rewards
        self.terminals = terminals

        self._ptr = len(observations) # pointer to the next position to be added
        self._size = len(observations)
     
    def normalize_obs(self, eps: float = 1e-3) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.observations.mean(0, keepdims=True)
        std = self.observations.std(0, keepdims=True) + eps
        self.observations = (self.observations - mean) / std
        self.next_observations = (self.next_observations - mean) / std
        obs_mean, obs_std = mean, std
        return obs_mean, obs_std

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:

        batch_indexes = np.random.randint(0, self._size, size=batch_size)
        
        return {
            "observations": torch.tensor(self.observations[batch_indexes]).to(self.device),
            "actions": torch.tensor(self.actions[batch_indexes]).to(self.device),
            "next_observations": torch.tensor(self.next_observations[batch_indexes]).to(self.device),
            "terminals": torch.tensor(self.terminals[batch_indexes]).to(self.device),
            "rewards": torch.tensor(self.rewards[batch_indexes]).to(self.device)
        }
    
    def sample_all(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations[:self._size].copy(),
            "actions": self.actions[:self._size].copy(),
            "next_observations": self.next_observations[:self._size].copy(),
            "terminals": self.terminals[:self._size].copy(),
            "rewards": self.rewards[:self._size].copy()
        }


class CQLPolicy():
    def __init__(self, actor, critic1, critic2, actor_optimizer, critic1_optimizer, critic2_optimizer, action_space, tau = 0.005, gamma = 0.99, alpha = 0.2, num_repeat_actions = 10):
        self.actor = actor
        self.critic1, self.critic1_old = critic1, deepcopy(critic1)
        self.critic1_old.eval()
        self.critic2, self.critic2_old = critic2, deepcopy(critic2)
        self.critic2_old.eval()
        
        self.actor_optimizer = actor_optimizer
        self.critic1_optimizer = critic1_optimizer
        self.critic2_optimizer = critic2_optimizer
        
        self.action_space = action_space
        
        self._tau = tau # for soft update
        self._gamma = gamma
        self._alpha = alpha
        
        self._num_repeat_actions = num_repeat_actions
        
    def train(self):
        self.actor.train()
        self.critic1.train()
        self.critic2.train()
        
    def eval(self):
        self.actor.eval()
        self.critic1.eval()
        self.critic2.eval()
        
    def soft_update(self):
        for param, target_param in zip(self.critic1.parameters(), self.critic1_old.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.critic2_old.parameters()):
            target_param.data.copy_(self._tau * param.data + (1.0 - self._tau) * target_param.data)
            
    def act_forward(self, observations):
        actor_distribution = self.actor(observations)
        squashed_action, raw_action = actor_distribution.rsample() ## squashed_action 是经过压缩后的动作向量，它已经被压缩到了一个有限的范围内。这个压缩过程通常是通过对采样结果应用一个双曲正切函数来实现的。而 raw_action 是未经过压缩的原始动作向量，它的值通常是任意实数。在 SAC 算法中，squashed_action 通常被用于执行环境中的动作，而 raw_action 则用于计算 actor loss 和 critic loss。
        log_prob = actor_distribution.log_prob(squashed_action, raw_action)
        return squashed_action, log_prob
    
    def select_action(self, observations):
        with torch.no_grad():
            squashed_action, _ = self.act_forward(observations)
        return squashed_action.cpu().numpy()
    
    def calculate_pi_values(self, observation_pi, observation_to_pred):
        action, log_prob = self.act_forward(observation_pi)
        
        q1 = self.critic1(observation_to_pred, action)
        q2 = self.critic2(observation_to_pred, action)
        
        return q1 - log_prob.detach(), q2 - log_prob.detach()
        
    def learn(self, batch):
        observations = batch["observations"]
        actions = batch["actions"]
        next_observations = batch["next_observations"]
        rewards = batch["rewards"]
        terminals = batch["terminals"]
        batch_size = len(terminals)
        
        # update critic 
        # q1 = self.critic1(observations, actions)
        # q2 = self.critic2(observations, actions)
        # with torch.no_grad():
        #     next_actions, next_log_prob = self.act_forward(next_observations)
        #     next_q1 = self.critic1_old(next_observations, next_actions)
        #     next_q2 = self.critic2_old(next_observations, next_actions)
        #     next_q = torch.min(next_q1, next_q2) - self._alpha * next_log_prob
        #     target_q = rewards + self._gamma * (1 - terminals) * next_q    
        
        # update actor
        actions, log_probs = self.act_forward(observations)
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        actor_loss = (self._alpha * log_probs - torch.min(q1, q2)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # compute td error
        with torch.no_grad():
            next_actions, next_log_probs = self.act_forward(next_observations)
            next_q = torch.min(
                self.critic1_old(next_observations, next_actions), 
                self.critic2_old(next_observations, next_actions)
            )
        
        target_q = rewards + self._gamma * (1 - terminals) * next_q
        q1 = self.critic1(observations, actions)
        q2 = self.critic2(observations, actions)
        critic1_loss = ((q1 - target_q).pow(2)).mean()
        critic2_loss = ((q2 - target_q).pow(2)).mean()
        
        # compute conservative loss
        random_actions = torch.FloatTensor(
            batch_size * self._num_repeat_actions, actions.shape[-1]
        ).uniform_(self.action_space.low[0], self.action_space.high[0]).to(self.actor.device)
        # tmp_obss & tmp_next_obss: (batch_size * num_repeat, obs_dim)
        tmp_observations = observations.unsqueeze(1).repeat(1, self._num_repeat_actions, 1).view(batch_size * self._num_repeat_actions, observations.shape[-1])
        tmp_next_observations = next_observations.unsqueeze(1).repeat(1, self._num_repeat_actions, 1).view(batch_size * self._num_repeat_actions, observations.shape[-1])
        
        observations_pi_value1, observations_pi_value2 = self.calculate_pi_values(tmp_observations, tmp_next_observations)
        next_observations_pi_value1, next_observations_pi_value2 = self.calculate_pi_values(tmp_next_observations, tmp_next_observations)
        
class MFPolicyTrainer:
    def __init__(self, policy, eval_env, buffer, epoch = 1000, step_per_epoch = 1000, batch_size = 256, eval_episodes = 10):
        self.policy = policy
        self.eval_env = eval_env
        self.buffer = buffer
        
        self._epoch = epoch
        self._step_per_epoch = step_per_epoch
        self._batch_size = batch_size
        self._eval_episodes = eval_episodes
        
    def train(self):
        start_time = time.time()
        num_timesteps = 0
        
        for epoch in range(1, self._epoch + 1):
            
            self.policy.train()

            pbar = tqdm(range(self._step_per_epoch), colour='green', desc=f'Epoch #{epoch}/{self._epoch}')
            for iteration in pbar:
                batch_sample = self.buffer.sample(self._batch_size)
                loss = self.policy.learn(batch_sample)
                pbar.set_postfix(**loss)
   
                
def train(args=get_args()):
    # create env and dataset
    env = gym.make(args.task)
    dataset = d4rl.qlearning_dataset(env)
    # See https://github.com/aviralkumar2907/CQL/blob/master/d4rl/examples/cql_antmaze_new.py#L22
    if 'antmaze' in args.task:
        dataset.rewards = (dataset.rewards - 0.5) * 4.0
    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    args.max_action = env.action_space.high[0]

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    env.seed(args.seed)

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=args.hidden_dims)
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=args.hidden_dims)
    distribution = TanhDiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )
    actor = ActorProb(actor_backbone, distribution, args.device)
    critic1 = Critic(critic1_backbone, args.device)
    critic2 = Critic(critic2_backbone, args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        target_entropy = args.target_entropy if args.target_entropy \
            else -np.prod(env.action_space.shape)

        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)
    else:
        alpha = args.alpha

    # create policy
    policy = CQLPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        tau=args.tau,
        gamma=args.gamma,
        alpha=alpha,
        # cql_weight=args.cql_weight,
        # temperature=args.temperature,
        # max_q_backup=args.max_q_backup,
        # deterministic_backup=args.deterministic_backup,
        # with_lagrange=args.with_lagrange,
        # lagrange_threshold=args.lagrange_threshold,
        # cql_alpha_lr=args.cql_alpha_lr,
        num_repeat_actions=args.num_repeat_actions
    )

    # create buffer
    buffer = ReplayBuffer(
        buffer_size=len(dataset["observations"]),
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32,
        device=args.device
    )
    buffer.load_dataset(dataset)

    # # log
    # log_dirs = make_log_dirs(args.task, args.algo_name, args.seed, vars(args))
    # # key: output file name, value: output handler type
    # output_config = {
    #     "consoleout_backup": "stdout",
    #     "policy_training_progress": "csv",
    #     "tb": "tensorboard"
    # }
    # logger = Logger(log_dirs, output_config)
    # logger.log_hyperparameters(vars(args))

    # create policy trainer
    policy_trainer = MFPolicyTrainer(
        policy=policy,
        eval_env=env,
        buffer=buffer,
        # logger=logger,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        batch_size=args.batch_size,
        eval_episodes=args.eval_episodes
    )

    # train
    policy_trainer.train()


if __name__ == "__main__":
    train()