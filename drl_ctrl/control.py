import json
import logging
import pathlib
from collections import deque

import numpy as np
import torch
import typing
import unityagents
import pandas as pd

from drl_ctrl import agents
from drl_ctrl import config as cfg
from drl_ctrl import path_util


def training(
        env: unityagents.UnityEnvironment,
        output_dir: typing.Union[pathlib.Path, str],
        agent_type: str = "DDPG",
        buffer_size: int = 100_000,
        batch_size: int = 128,
        gamma_discount_factor: float = 0.95,
        tau_soft_update: float = 1e-3,
        learning_rate_actor: float = 2e-3,
        learning_rate_critic: float = 1e-3,
        l2_weight_decay: float = 0.0,
        update_every: int = 10,
        num_updates: int = 20,
        has_ou_noise: bool = True,
        ou_noise_mu: float = 0.0,
        ou_noise_theta: float = 0.15,
        ou_noise_sigma: float = 0.1,
        n_episodes: int = 500,
        mean_score_threshold: float = 30.0,
        max_t: int = 1000,
        agent_seed=111_111,
        logging_freq: int = 10):
    """
    Train agent for Unity Reacher environment and save results.

    Train a deep reinforcement learning agent (Reacher) or 20 agents (Reacher20)
    to reach surfaces of spheres with a two joint arm in Unity Environment and
    save results (training scores, agent neural network model weights,
    metadata with hyper-parameters) to provided output directory.

    Parameters
    ----------
    env
        Unity environment
    output_dir
        Path to output results output directory (scores, weights, metadata)
    agent_type
        A type of agent to train from the available ones
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
    update_every
        Update weights of networks every `update_every` time steps
    num_updates
        Number of simultaneous updates
    has_ou_noise
        If True, Ornstein-Uhlenbeck noise is added to actions
    ou_noise_mu
        Ornstein-Uhlenbeck process mu parameter
    ou_noise_theta
        Ornstein-Uhlenbeck process theta parameter
    ou_noise_sigma
        Ornstein-Uhlenbeck process sigma parameter
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t:
        Maximum number of time steps per episode
    agent_seed
        Random seed for agent epsilon-greedy policy
    logging_freq
        Logging frequency

    """
    logger = logging.getLogger(__name__)

    output_dir = pathlib.Path(output_dir)

    logger.info(f"Ensuring output directory exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    path_weights_actor = path_util.mk_path_weights_actor(output_dir)
    path_weights_critic = path_util.mk_path_weights_critic(output_dir)
    path_scores = path_util.mk_path_scores(output_dir)
    path_metadata = path_util.mk_path_metadata(output_dir)

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=True)[brain_name]
    state = env_info.vector_observations[0]
    state_size = len(state)

    agent = agents.DDPGAgent(
        state_size=state_size,
        action_size=action_size,
        num_agents=len(env_info.agents),
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma_discount_factor=gamma_discount_factor,
        tau_soft_update=tau_soft_update,
        learning_rate_actor=learning_rate_actor,
        learning_rate_critic=learning_rate_critic,
        l2_weight_decay=l2_weight_decay,
        update_network_every=update_every,
        num_updates=num_updates,
        ou_noise_mu=ou_noise_mu,
        ou_noise_theta=ou_noise_theta,
        ou_noise_sigma=ou_noise_sigma,
        seed=agent_seed)

    scores = train_agent(
        env=env,
        agent=agent,
        n_episodes=n_episodes,
        mean_score_threshold=mean_score_threshold,
        max_t=max_t,
        has_ou_noise=has_ou_noise,
        logging_freq=logging_freq)

    logger.info(f'Saving actor network model weights to {str(path_weights_actor)}')
    torch.save(agent.actor_local.state_dict(), str(path_weights_actor))
    logger.info(f'Actor model weights saved successfully!')

    logger.info(f'Saving critic network model weights to {str(path_weights_critic)}')
    torch.save(agent.critic_local.state_dict(), str(path_weights_critic))
    logger.info(f'Critic model weights saved successfully!')

    logger.info(f'Saving training scores to {str(path_scores)}')
    scores_df = pd.DataFrame.from_records(
        enumerate(scores, start=1),
        columns=(cfg.SCORE_COLNAME_X, cfg.SCORE_COLNAME_Y))
    logger.info(f'Training scores saved successfully!')

    scores_df.to_csv(path_scores, index=False)

    logger.info(f'Saving training metadata to {str(path_metadata)}')
    metadata = {
        "agent_type": agent_type,
        "agent": agent.metadata,
        "mean_score_threshold": mean_score_threshold,
        "max_t": max_t,
        "has_ou_noise": has_ou_noise,
    }

    with open(path_metadata, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f'Training metadata saved successfully!')


def train_agent(
        env: unityagents.UnityEnvironment,
        agent: agents.DDPGAgent,
        n_episodes: int = 200,
        mean_score_threshold: float = 30.0,
        max_t: int = 1000,
        has_ou_noise: bool = True,
        logging_freq: int = 10,
        scores_maxlen: int = 100) -> typing.List[float]:
    """
    Train agent for Unity Reacher environment and return scores.

    Parameters
    ----------
    env
        Unity environment
    agent
        And instance of Deep Reinforcement Learning Agent from drl_ctrl.agents module
    n_episodes
        Maximum number of episodes
    mean_score_threshold
        Threshold of mean last 100 weights to stop training and save results
    max_t
        Maximum number of time steps per episode
    has_ou_noise
        If True, Ornstein-Uhlenbeck noise is added to actions
    logging_freq
        Logging frequency
    scores_maxlen
        Maximum length of scores window

    """

    logger = logging.getLogger(__name__)

    scores = []
    scores_window = deque(maxlen=scores_maxlen)

    for i_episode in range(1, n_episodes+1):
        brain_name = env.brain_names[0]
        env_info = env.reset(train_mode=True)[brain_name]
        agent.reset()
        states = env_info.vector_observations
        num_agents = len(env_info.agents)
        agent_scores = np.zeros(num_agents)

        for t in range(max_t):
            # choose action (for each agent)
            actions = agent.act(states, add_noise=has_ou_noise)

            # take action in the environment(for each agent)
            env_info = env.step(actions)[brain_name]

            # get next state (for each agent)
            next_states = env_info.vector_observations

            # see if episode finished
            dones = env_info.local_done

            # update the score (for each agent)
            agent_scores += env_info.rewards

            agent.step(states, actions, env_info.rewards, next_states, dones)

            # roll over states to next time step
            states = next_states

            # exit loop if episode finished
            if np.any(dones):
                break

        score = float(np.mean(agent_scores))
        scores_window.append(score)
        scores.append(score)

        if i_episode % logging_freq == 0:
            logger.info(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')

        if len(scores_window) == scores_maxlen and np.mean(scores_window) >= mean_score_threshold:
            logger.info(
                f'\nEnvironment solved in {i_episode-100:d} episodes!'
                f'\tAverage Score: {np.mean(scores_window):.2f}')
            break

    return scores


def demo(
        env: unityagents.UnityEnvironment,
        dir_model: typing.Optional[pathlib.Path] = None
) -> float:
    """
    Run a demo on the environment

    Parameters
    ----------
    env
        Unity Environment
    dir_model
        If provided, agent model weights are loaded from path,
        Random Agent is used otherwise

    Returns
    -------
    float
        final score

    """
    if dir_model is not None:
        brain_name = env.brain_names[0]
        brain = env.brains[brain_name]
        action_size = brain.vector_action_space_size
        env_info = env.reset(train_mode=False)[brain_name]
        state = env_info.vector_observations[0]
        state_size = len(state)

        agent = agents.DDPGAgent(state_size=state_size, action_size=action_size, num_agents=20)
        agent.actor_local.load_state_dict(torch.load(path_util.mk_path_weights_actor(dir_model)))
        agent.critic_local.load_state_dict(torch.load(path_util.mk_path_weights_critic(dir_model)))

        return demo_trained(env, agent)
    else:
        return demo_random(env)


def demo_trained(env: unityagents.UnityEnvironment, agent: agents.DDPGAgent) -> float:
    """
    Run a demo of a trained agent

    Parameters
    ----------
    env
        Unity Environment
    agent
        trained agent

    Returns
    -------
    float
        final score

    """

    brain_name = env.brain_names[0]
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    while True:
        actions = agent.act(states)
        # send all actions to the environment
        env_info = env.step(actions)[brain_name]
        # get next state (for each agent)
        next_states = env_info.vector_observations
        # see if episode finished
        dones = env_info.local_done
        # update the score (for each agent)
        scores += env_info.rewards
        # roll over states to next time step
        states = next_states
        # exit loop if episode finished
        if np.any(dones):
            break

    score = float(np.mean(scores))
    print('Total score (averaged over agents) this episode: {}'.format(score))
    return score


def demo_random(env: unityagents.UnityEnvironment) -> float:
    """
    Run a demo of a Random Agent

    Parameters
    ----------
    env
        Unity Environment

    Returns
    -------
    float
        final score

    """
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    action_size = brain.vector_action_space_size

    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    num_agents = len(env_info.agents)
    scores = np.zeros(num_agents)
    i_step = 1
    while True:
        actions = np.random.randn(num_agents, action_size)  # select an action (for each agent)
        actions = np.clip(actions, -1, 1)  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]  # send all actions to the environment
        next_states = env_info.vector_observations  # get next state (for each agent)
        dones = env_info.local_done  # see if episode finished
        scores += env_info.rewards  # update the score (for each agent)
        states = next_states  # roll over states to next time step
        if np.any(dones):  # exit loop if episode finished
            print(f'Episode finished in {i_step} steps')
            break
        i_step += 1

    score = float(np.mean(scores))
    print('Total score (averaged over agents) this episode: {}'.format(score))
    return score
