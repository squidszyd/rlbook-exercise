import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from typing import Dict, Tuple


class Env:

  def __init__(self, mu=-0.1, var=1.0, num_actions_at_b: int = 10):
    self._num_actions_at_b = num_actions_at_b
    # 0: left, 1: right
    self._valid_actions = {
        0: (0, 1),
        1: tuple(range(num_actions_at_b)),
        2: (0,)
    }
    transition_at_b = {
        k: (2, lambda: np.random.normal(mu, var))
        for k in range(num_actions_at_b)
    }
    self._transition_map = {
        0: {
            0: (1, lambda: 0.0),
            1: (2, lambda: 0.0)
        },
        1: transition_at_b,
    }
    self._state = 0

  @property
  def valid_actions(self):
    return self._valid_actions

  def reset(self):
    self._state = 0
    return self._state

  def step(self, action: int) -> Tuple[int, bool, float]:
    trans = self._transition_map[self._state]
    if action not in trans:
      raise f'Action-{action} is not valid at state-{self._state}'
    observation, reward_fn = trans[action]
    reward = reward_fn()
    self._state = observation
    return (observation, observation == 2, reward)


class AgentBase:

  def sample_action(self, state: int, greedy: bool = False) -> int:
    raise NotImplementedError

  def learn(self, observation: int, reward: float, action: int, state: int):
    pass


class QAgent(AgentBase):

  def __init__(self,
               valid_actions: Dict[int, Tuple[int]],
               epsilon: float = 0.1,
               alpha: float = 0.1):
    self._q = {
        k: np.zeros(shape=(len(v),), dtype=float)
        for k, v in valid_actions.items()
    }
    self._eps = epsilon
    self._alpha = alpha

  def sample_action(self, state: int, greedy: bool = False) -> int:
    q = self._q[state]
    a = np.random.choice(np.where(q == np.max(q))[0])
    # print(f'S={state}, Q[{state}]={q}, A={a}')
    if not greedy and np.random.sample() < self._eps:
      a = np.random.choice(len(q))
    return a

  def learn(self, observation: int, reward: float, action: int, state: int):
    update = reward + np.max(self._q[observation]) - self._q[state][action]
    self._q[state][action] = self._q[state][action] + self._alpha * update
    return


class DoubleQAgent(AgentBase):

  def __init__(self,
               valid_actions: Dict[int, Tuple[int]],
               epsilon: float = 0.1,
               alpha: float = 0.1):
    self._q1 = {
        k: np.zeros(shape=(len(v),), dtype=float)
        for k, v in valid_actions.items()
    }
    self._q2 = {
        k: np.zeros(shape=(len(v),), dtype=float)
        for k, v in valid_actions.items()
    }
    self._eps = epsilon
    self._alpha = alpha

  def sample_action(self, state: int, greedy: bool = False) -> int:
    q = self._q1[state] + self._q2[state]
    a = np.random.choice(np.where(q == np.max(q))[0])
    if not greedy and np.random.sample() < self._eps:
      a = np.random.choice(len(q))
    return a

  def learn(self, observation: int, reward: float, action: int, state: int):
    if np.random.sample() < 0.5:
      a = np.argmax(self._q2[observation])
      update = reward + self._q1[observation][a] - self._q1[state][action]
      self._q1[state][action] += self._alpha * update
    else:
      a = np.argmax(self._q1[observation])
      update = reward + self._q2[observation][a] - self._q2[state][action]
      self._q2[state][action] += self._alpha * update


def one_episode(env: Env, agent: AgentBase, greedy: bool = False):
  t, total_reward = 0, 0.0
  state = env.reset()
  num_left_at_a = 0
  while True:
    action = agent.sample_action(state, greedy)
    num_left_at_a += (1 if state == 0 and action == 0 else 0)
    observation, done, reward = env.step(action)
    agent.learn(observation, reward, action, state)
    state = observation
    t += 1
    total_reward += reward
    if done:
      break

  return (t, total_reward, num_left_at_a)


def loop(env: Env, agent: AgentBase, num_episodes: int = 300):
  num_left_at_a = np.zeros(shape=(num_episodes,), dtype=float)
  for i in range(num_episodes):
    _, _, num = one_episode(env, agent)
    num_left_at_a[i] += num
  return num_left_at_a


if __name__ == "__main__":
  env = Env()

  num_exps = 2000
  num_episodes = 300
  num_left_at_a_q = np.zeros(shape=(num_episodes,), dtype=float)
  num_left_at_a_d = np.zeros(shape=(num_episodes,), dtype=float)

  for exp in tqdm(range(num_exps)):
    q_agent = QAgent(env.valid_actions, 0.1, 0.1)
    num_left_at_a_q += loop(env, q_agent, num_episodes)

    d_agent = DoubleQAgent(env.valid_actions, 0.1, 0.1)
    num_left_at_a_d += loop(env, d_agent, num_episodes)

  num_left_at_a_q /= num_exps
  num_left_at_a_d /= num_exps

  xs = list(range(num_episodes))
  ys_q = num_left_at_a_q.tolist()
  ys_d = num_left_at_a_d.tolist()
  plt.plot(xs, ys_q, 'g-', label='q-learning')
  plt.plot(xs, ys_d, 'r-', label='double q-learning')
  plt.xlabel('episodes')
  plt.ylabel('% left from A')
  plt.legend()

  plt.show()
