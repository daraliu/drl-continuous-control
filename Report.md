# Report

This repository contains a simple implementation of [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971)
and the code required to train it for Unity Reacher environment with 20 agents. 

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The environment is considered solved if the average score of the last 100 episodes is above 30.0. Here we solve the environment with 20 agents, therefor the score for an episode is the average score of all 20 agents.


## Learning Algorithm

Deep Deterministic Policy Gradients with implementation that works for multiple agents.


### Hyperparameters

The following hyper parameters were used for training.

#### Common hyper parameters for all agents

| parameter            | value |
| -------------------- | ----- |
| n_episodes           | 500   |
| mean_score_threshold | 30.0  |
| max_t                | 1000  |
| learning_rate_actor  | 0.002 |
| learning_rate_critic | 0.001 |
| tau_soft_update      | 0.001 |
| l2_weight_decay      | 0     |
| has_ou_noise         | True  |
| ou_noise_mu          | 0.0   |
| ou_noise_theta       | 0.15  |
| agent_seed           | 11111 |
| logging_freq         | 1     |


#### Varying hyper parameters for different agents

| agent_id | batch_size | buffer_size | gamma_discount_factor | num_updates | ou_noise_sigma | update_every | n_episodes_to_solve |
| -------- | ---------- | ----------- | --------------------- | ----------- | -------------- | ------------ | ------------------- |
| 0        | 128        | 100000      | 0.99                  | 20          | 0.2            | 10           | 14                  |
| 1        | 128        | 100000      | 0.95                  | 20          | 0.2            | 10           | 2                   |
| 2        | 128        | 100000      | 0.95                  | 20          | 0.1            | 10           | 0                   |
| 3        | 1024       | 1000000     | 0.95                  | 20          | 0.1            | 10           | 0                   |
| 4        | 128        | 100000      | 0.99                  | 10          | 0.2            | 20           | 400                 |
| 5        | 1024       | 1000000     | 0.95                  | 20          | 0.2            | 10           | 0                   |
| 6        | 128        | 100000      | 0.99                  | 10          | 0.1            | 20           | 126                 |
| 7        | 128        | 100000      | 0.95                  | 20          | 0.1            | 10           | 2                   |
| 8        | 1024       | 1000000     | 0.99                  | 20          | 0.2            | 10           | 0                   |
| 9        | 128        | 100000      | 0.99                  | 20          | 0.1            | 10           | 0                   |
| 10       | 128        | 100000      | 0.95                  | 10          | 0.2            | 20           | 400                 |
| 11       | 128        | 100000      | 0.95                  | 10          | 0.1            | 20           | 50                  |

There is a trade-off with `batch_size` parameter. `batch_size=1024` increase convergence significantly (see Plot of Rewards), but increase the wall-clock time required for training due to heavier gradient computations involved when compared to `batch_size=128`.

### Neural Network Model Architectures

Actor Neural Network with 3 fully connected hidden layers and batch normalization:

- fc1, in:`state_size`, out:128, relu activation
- Batch Normalization
- fc2, in: 128, out:128, relu activation
- fc3, in: 128, out: `action_size`, _tahn_ activation

here `state_size=33`, `action_size=4`.

### Critic Network Model Architectures

Critic Neural Network with 3 fully connected hidden layers and batch normalization:

- fcs1, in:`state_size`, out:128, relu activation
- Batch Normalization
- fc2, in: 128+`action_size`, out:128, relu activation
- fc3, in: 128, out: 1

here `state_size=33`, `action_size=4`, `output_size=1`

## Plot of Rewards

Score for all agents

![](https://github.com/daraliu/drl-continuous-control/blob/master/training_output/tuning_results/scores_all.png)

Zoomed in plot of rewards for best agents that solved the environment faster than 5 (105) episodes.

![](https://github.com/daraliu/drl-continuous-control/blob/master/training_output/tuning_results/scores_best.png)

Fastest training agent by episode.

![](https://github.com/daraliu/drl-continuous-control/blob/master/img/best_score_so_far.png)


Training of the agent done in [Reacher20-Continuous-Control.ipynb](https://github.com/daraliu/drl-continuous-control/blob/master/notebooks/Reacher20-Continuous-Control.ipynb) Jupyter notebook.

Environment was solved in 0 (100) episodes with multiple hyper-parameter sets. The best solution achieves average of  is 36.17 at 0 (100) episode.

![](https://github.com/daraliu/drl-continuous-control/blob/master/img/best_agent_so_far.png)

## Ideas for Future Work

To improve agent performance, the following steps should be taken:
- Try [Proximal Policy Optimization (PPO)](https://arxiv.org/pdf/1707.06347.pdf) algorithm for Reacher environment.
- Try [Asynchronous Advantage Actor-Critic (A3C)](https://arxiv.org/pdf/1602.01783.pdf) algorithm for Reacher environment.
- Try Advantage Actor-Critic (A2C) algorithm for Reacher environment.
- Try [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://openreview.net/pdf?id=SyZipzbCb) algorithm for Reacher environment.
- Experiment with more Neural Network architectures - evaluate trade-off between simpler networks for faster learning versus complex networks for greater score.
- Perform more thorough hyper parameter turing and analysis - multiple runs per hyper parameter set to evaluate their stability, do more exploration in hyper parameter space to draw better conclusions DDPG and Reacher environment.
