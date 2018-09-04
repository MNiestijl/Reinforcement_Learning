import numpy as np
import random
from collections import deque


class MDPAgent():

	def __init__(self, env, policy, observation=None, max_timesteps=1000, eps=0, eps_decay=0.99, memory_len=int(1e3), train_n_samples=1):
		self.policy = policy
		self.env = env
		self.observation = observation if observation else env.reset()
		self.max_timesteps = max_timesteps
		self.eps = eps
		self.eps_decay = eps_decay
		self.train_n_samples=train_n_samples
		self.experiences = deque(maxlen=memory_len)

	def _get_action(self, train=True):
		# During training, take epsilon-greedy action.
		if train:
			take_random_action =  random.choices([True, False], weights=[self.eps,1-self.eps])[0]
			if take_random_action:
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

	def __perform_episode(self,train=True, render=False):
		total_reward = 0
		self.observation = self.env.reset()
		for timestep in range(self.max_timesteps):

			if render:
				self.env.render()
			action = self._get_action(train=train)
			observation, reward, done, info  = self.env.step(action)
			new_experience = (self.observation, action, observation, reward, done)
			if train:
				# For now, we only use experiences gained during a training phase
				self.experiences.append(new_experience)
				train_experiences = random.choices(list(self.experiences), k=self.train_n_samples)
				self.policy.train_on_batch(train_experiences)
				self.eps *= self.eps_decay
			self.observation = observation
			total_reward += reward
			if done:
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
