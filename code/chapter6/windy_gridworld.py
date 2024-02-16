import gym
import gym_gridworlds
import numpy as np
import argparse
from tqdm import tqdm
from typing import Tuple, SupportsFloat


class Agent(object):

  def __init__(
      self,
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

  def choose_action(self, state: Tuple[int, int], greedy: bool = False) -> int:
    q_s = self._q[state[0], state[1]]
    a = np.argmax(q_s)
    if not greedy and np.random.random_sample() < self._epsilon:
      a = np.random.choice(self._action_size)
    return a

  def learn(
      self,
      observation: Tuple[int, int],
      reward: SupportsFloat,
      state: Tuple[int, int],
      action: int,
  ) -> None:
    a1 = self.choose_action(observation)
    a0 = action

    q1 = self._q[observation[0], observation[1], a1]
    q0 = self._q[state[0], state[1], a0]

    update = self._alpha * (reward + q1 - q0)
    self._q[state[0], state[1], a0] = q0 + update

if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--epsilon', type=float, default=0.1, help='epsilon-greedy')
  p.add_argument('--learning-rate',
                 type=float,
                 default=0.5,
                 help='learning rate')
  p.add_argument('--max-episodes', type=int, default=8000, help='max episodes')
  a = p.parse_args()

  env = gym.make('WindyGridworld-v0')

  agent = Agent(
      env_shape=(7, 10),
      action_size=4,
      epsilon=a.epsilon,
      alpha=a.learning_rate,
  )

  for i in tqdm(range(a.max_episodes)):
    done = False
    state, _ = env.reset()
    t = 0
    total_reward = 0
    while not done:
      action = agent.choose_action(state)
      (observation, reward, done, _) = env.step(action)
      agent.learn(observation=observation,
                  reward=reward,
                  state=state,
                  action=action)
      state = observation
      t += 1
      total_reward += reward
      if done:
        print(f'Episode-{i} finished after {t} timesteps, total reward: {total_reward}')
        t = 0
        total_reward = 0
        break
  
  # plot learnt policy
  v_star = np.zeros(shape=(7, 10), dtype=int)
  state, _ = env.reset()
  done = False
  t = 0
  total_reward = 0
  while not done:
    action = agent.choose_action(state, greedy=True)
    observation, reward, done, _ = env.step(action)
    v_star[observation[0], observation[1]] = 1
    state = observation
    t += 1
    total_reward += reward
    if done:
      print(f'Optimal policy takes {t} timesteps, total reward: {total_reward}')
      state, _ = env.reset()
      break

  print(v_star)