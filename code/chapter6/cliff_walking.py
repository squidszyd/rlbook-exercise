import gym
import gym_gridworlds
import numpy as np
import argparse

from typing import SupportsFloat, Tuple


class AgentBase(object):

  def __init__(
      self,
      name: str,
      env_shape: Tuple[int, int],
      action_size: int,
      epsilon: SupportsFloat = 0.1,
      alpha: SupportsFloat = 0.5,
  ):
    self._q = np.zeros(
        shape=(env_shape[0], env_shape[1], action_size),
        dtype=float,
    )
    self._action_size = action_size
    self._epsilon = epsilon
    self._alpha = alpha
    self._name = name

  @property
  def name(self):
    return self._name

  def choose_action(self, state: Tuple[int, int], greedy: bool = False):
    q_s = self._q[state[0], state[1]]
    a = np.argmax(q_s)
    if not greedy and np.random.sample() < self._epsilon:
      a = np.random.choice(self._action_size)
    return a

  def learn(
      self,
      observation: Tuple[int, int],
      reward: SupportsFloat,
      state: Tuple[int, int],
      action: int,
  ) -> None:
    raise NotImplementedError


class SarsaAgent(AgentBase):

  def __init__(self,
               name: str,
               env_shape: Tuple[int, int],
               action_size: int,
               epsilon: SupportsFloat = 0.1,
               alpha: SupportsFloat = 0.5):
    super().__init__(name, env_shape, action_size, epsilon, alpha)

  def learn(
      self,
      observation: Tuple[int, int],
      reward: SupportsFloat,
      state: Tuple[int, int],
      action: int,
  ) -> None:
    a1 = self.choose_action(observation)
    q1 = self._q[observation[0], observation[1], a1]

    a0 = action
    q0 = self._q[state[0], state[1], a0]

    update = self._alpha * (reward + q1 - q0)
    self._q[state[0], state[1], a0] += update


class QAgent(AgentBase):

  def __init__(self,
               name: str,
               env_shape: Tuple[int, int],
               action_size: int,
               epsilon: SupportsFloat = 0.1,
               alpha: SupportsFloat = 0.5):
    super().__init__(name, env_shape, action_size, epsilon, alpha)

  def learn(
      self,
      observation: Tuple[int, int],
      reward: SupportsFloat,
      state: Tuple[int, int],
      action: int,
  ) -> None:
    a0 = action
    q0 = self._q[state[0], state[1], a0]

    update = self._alpha * (
        reward + np.max(self._q[observation[0], observation[1]]) - q0)
    self._q[state[0], state[1], a0] += update


def train(env, agent: AgentBase, max_episodes: int):
  avg_reward = 0
  for i in range(max_episodes):
    t = 0
    total_reward = 0
    state, _ = env.reset()
    while True:
      action = agent.choose_action(state)
      (observation, reward, done, _) = env.step(action)
      agent.learn(observation, reward, state, action)
      state = observation
      t += 1
      total_reward += reward
      if done:
        # print(
        #     f'Agent-{agent.name} Episode-{i}: finished after {t} timesteps, total reward: {total_reward}'
        # )
        state, _ = env.reset()
        avg_reward += total_reward
        t = 0
        total_reward = 0
        break

  avg_reward /= max_episodes
  print(f'Agent-{agent.name} average train reward: {avg_reward}')


def print_optimal_path(env, agent: AgentBase):
  state, _ = env.reset()
  path = np.zeros(shape=(4, 12), dtype=int)
  path[state[0], state[1]] = 1
  t = 0
  total_reward = 0
  while True:
    action = agent.choose_action(state, True)
    (observation, reward, done, _) = env.step(action)
    path[observation[0], observation[1]] = 1
    state = observation
    t += 1
    total_reward += reward
    if done:
      print(
          f'Agent-{agent.name} finished after {t} timesteps, total reward: {total_reward}'
      )
      print(f'Agent-{agent.name} optimal path:\n{path}')
      break


if __name__ == "__main__":
  p = argparse.ArgumentParser()
  p.add_argument('--epsilon', type=float, default=0.1)
  p.add_argument('--learning-rate', type=float, default=0.5)
  p.add_argument('--max-episodes', type=int, default=8000)
  a = p.parse_args()

  env = gym.make('Cliff-v0')

  sarsa_agent = SarsaAgent(
      name='sarsa',
      env_shape=(4, 12),
      action_size=4,
      epsilon=a.epsilon,
      alpha=a.learning_rate,
  )
  train(env, sarsa_agent, a.max_episodes)
  print_optimal_path(env, sarsa_agent)

  q_agent = QAgent(
      name='Q',
      env_shape=(4, 12),
      action_size=4,
      epsilon=a.epsilon,
      alpha=a.learning_rate,
  )
  train(env, q_agent, a.max_episodes)
  print_optimal_path(env, q_agent)
