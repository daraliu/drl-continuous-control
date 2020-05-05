import random
import copy
from collections import deque
from typing import Iterable, Any, List, NamedTuple, Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from drl_ctrl.model import ActorNetwork, CriticNetwork


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPGAgent:
    """Interacts with and learns from the environment."""
    
    def __init__(
            self,
            state_size: int,
            action_size: int,
            num_agents: int,
            buffer_size: int = 100_000,
            batch_size: int = 128,
            gamma_discount_factor: float = 0.95,
            tau_soft_update: float = 1e-3,
            learning_rate_actor: float = 2e-3,
            learning_rate_critic: float = 1e-3,
            l2_weight_decay: float = 0.0,
            update_network_every: int = 10,
            num_updates: int = 20,
            ou_noise_mu: float = 0.0,
            ou_noise_theta: float = 0.15,
            ou_noise_sigma: float = 0.1,
            seed: int = 0):
        """

        Parameters
        ----------
        state_size
            Size of state space
        action_size
            Size of action space
        buffer_size
            Maximum size of buffer for storing experiences
        batch_size
            Size of Each training batch
        gamma_discount_factor
            Discount factor
        tau_soft_update
            Interpolation parameter for soft network weight update
        learning_rate_actor
            Learning rate for Actor network
        learning_rate_critic
            Learning rate for Critic network
        l2_weight_decay
            Weight decay for critic optimizer
        update_network_every
            Update network weight every `update_network_every` time steps
        num_updates
            Number of simultaneous updates
        ou_noise_mu
            Ornstein-Uhlenbeck process mu parameter
        ou_noise_theta
            Ornstein-Uhlenbeck process theta parameter
        ou_noise_sigma
            Ornstein-Uhlenbeck process sigma parameter
        seed
            Random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma_discount_factor = gamma_discount_factor
        self.tau_soft_update = tau_soft_update
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.l2_weight_decay = l2_weight_decay
        self.update_network_every = update_network_every
        self.num_updates = num_updates
        self.ou_noise_mu = ou_noise_mu
        self.ou_noise_theta = ou_noise_theta
        self.ou_noise_sigma = ou_noise_sigma
        self.seed = random.seed(seed)

        # Actor Network (w/ Target Network)
        self.actor_local = ActorNetwork(state_size, action_size, seed).to(DEVICE)
        self.actor_target = ActorNetwork(state_size, action_size, seed).to(DEVICE)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=learning_rate_actor)

        # Critic Network (w/ Target Network)
        self.critic_local = CriticNetwork(state_size, action_size, seed).to(DEVICE)
        self.critic_target = CriticNetwork(state_size, action_size, seed).to(DEVICE)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(),
            lr=learning_rate_critic,
            weight_decay=l2_weight_decay)

        # Noise process
        self.noise = OUNoise(
            (num_agents, action_size),
            seed, ou_noise_mu, ou_noise_theta, ou_noise_sigma)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'action_size': self.action_size,
            'state_size': self.state_size,
            'buffer_size': self.buffer_size,
            'batch_size': self.batch_size,
            'gamma_discount_factor': self.gamma_discount_factor,
            'tau_soft_update': self.tau_soft_update,
            'learning_rate_actor': self.learning_rate_actor,
            'learning_rate_critic': self.learning_rate_critic,
            'l2_weight_decay': self.l2_weight_decay,
            'update_network_every': self.update_network_every,
            'num_updates': self.num_updates,
            'actor_local': self.actor_local.metadata,
            'critic_local': self.critic_local.metadata,
            'ou_noise_mu': self.ou_noise_mu,
            'ou_noise_theta': self.ou_noise_theta,
            'ou_noise_sigma': self.ou_noise_sigma,
            'seed': self.seed,
        }

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        self.memory.add_batch(states, actions, rewards, next_states, dones)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_network_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                # Apply several simultaneous updates for actor and critic networks
                for _ in range(self.num_updates):
                    experiences = self.memory.sample()
                    self.learn(experiences, self.gamma_discount_factor)

    def act(self, states, add_noise=True):
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to(DEVICE)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        self.noise.reset()
        self.t_step = 0

    def learn(
            self,
            experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            gamma: float):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Parameters
        ----------
        experiences
            tuple of (s, a, r, s', done) tuples
        gamma
            discount factor

        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss with gradient clipping
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        _soft_update(self.critic_local, self.critic_target, self.tau_soft_update)
        _soft_update(self.actor_local, self.actor_target, self.tau_soft_update)


def _soft_update(local_model, target_model, tau):
    """Soft update model parameters.
    θ_target = τ*θ_local + (1 - τ)*θ_target

    Parameters
    ----------
    local_model : PyTorch model
        weights will be copied from
    target_model : PyTorch model
        weights will be copied to
    tau : float
        interpolation parameter
    """
    for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
        target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, shape, seed, mu=0., theta=0.15, sigma=0.1):
        """Initialize parameters and noise process."""
        self.state = None
        self.mu = mu * np.ones(shape)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(*x.shape)
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            seed: int):
        """Initialize a ReplayBuffer object.

        Parameters
        ----------
        action_size
            dimension of each action
        buffer_size
            maximum size of buffer
        batch_size
            size of each training batch
        seed
            random seed
        """

        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def add_batch(self, states, actions, rewards, next_states, dones):
        """Add new experiences batch to memory."""
        experiences = zip(states, actions, rewards, next_states, dones)
        self.memory.extend([Experience(*e) for e in experiences])

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(_drop_none(states))).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(_drop_none(actions))).float().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(_drop_none(rewards))).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(_drop_none(next_states))).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(_drop_none(dones)).astype(np.uint8)).float().to(DEVICE)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def _drop_none(xs: Iterable[Any]) -> List[Any]:
    return [x for x in xs if x is not None]


class Experience(NamedTuple):
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool
