import numpy as np
import random
from collections import deque


class MDPAgent():
	"""
	Currently, this makes use of "experience-replay" and therefore does not support on-line algorithms such as actor-cricit algorithms. Off-line algorithms are supported.
	"""

	def __init__(self, env, policy, observation=None, max_timesteps=1000, eps=0.01 , eps_decay=0.99, memory_len=int(1e3), train_n_samples=100, n_steps=100):
		self.policy = policy
		self.env = env
		self.observation = observation if observation else env.reset()
		self.max_timesteps = max_timesteps
		self.eps = eps
		self.eps_decay = eps_decay
		self.train_n_samples=train_n_samples
		self.n_steps = n_steps # perform n_steps inbetween parameter-fitting sessions. If n_steps=-1, then perform full-episodes.
		self.experiences = deque(maxlen=memory_len)
		self.counter = 1

	def _get_action(self, train=True):
		# During training, take epsilon-greedy action.
		if train:

			if np.random.choice([True, False], p=[self.eps,1-self.eps]):
					action = self.env.action_space.sample()
			else:	
				try: 
					# During training, use stochastic action if possible
					action = self.policy.get_stochastic_action(self.observation)
				except NotImplementedError:
					action = self.policy.get_action(self.observation)
		else:
			action = self.policy.get_action(self.observation)
		return action

	def _update(self):
		train_experiences = random.choices(list(self.experiences), k=self.train_n_samples)
		self.policy.train_on_batch(train_experiences)
		self.eps *= self.eps_decay

	def __perform_episode(self,train=True, render=False):
		total_reward = 0
		self.observation = self.env.reset()
		for _ in range(self.max_timesteps):
			if render:
				self.env.render()
			action = self._get_action(train=train)
			observation, reward, done, info  = self.env.step(action)
			self.experiences.append((self.observation, action, observation, reward, done))
			if train and not self.n_steps==-1 and self.counter%self.n_steps==0:
				self._update()
				self.counter = 0
			self.observation = observation
			total_reward += reward
			self.counter += 1
			if done:
				if self.n_steps==-1:
					self._update()
				break
		return total_reward

	def train(self, n_episodes=1, render_episodes=0):
		"""
		render_episodes: number of episodes in between rendering. Zero (default) corresponds to not rendering at all.
		"""
		func = lambda i_episode: self.__perform_episode(render=(render_episodes>0) and (i_episode % render_episodes==0))
		return list(map(func, range(n_episodes)))
			
	def scores(self, n_episodes, render_episodes=0):
		func = lambda i_episode: self.__perform_episode(train=False, render=(render_episodes>0) and (i_episode % render_episodes==0))
		return list(map(func, range(n_episodes)))
