import numpy as np
import numpy.random as rnd
from copy import copy

class ES():
	"""
	Paper: https://arxiv.org/abs/1703.03864
	TODO: 
	- Implement parallelization
	- Implement a second order solver, i.e., estimate the hessian matrix stochastically and implement the saddle-free newtons method 
	to hopefully boost performance.
	"""

	def __init__(self, env, policy, learning_rate, sigma, observation=None, max_timesteps=100):
		"""
		policy: Function or random variable S -> A depending on some vector of real valued parameters theta.
		sigma: noise variance
		"""
		self.policy = policy
		self.params = copy(policy.params)
		self.env = env
		self.observation = observation if observation else env.reset()
		self.max_timesteps = max_timesteps

	def __perform_episode(self, train=True, render=False):
		#
		"""
		TODO: This can be done in parallel.
		"""

		if train:
			self.policy.params = self.params + self.sigma*eps

		# Compute returns
		total_reward = 0
		self.observation = self.env.reset()
		for timestep in range(self.max_timesteps):
			if render:
				self.env.render()
			self.policy.get_action(self.observation)
			self.observation, reward, done, info = self.env.step(action)
			total_reward += reward
			if done:
				break
		return total_reward

	def train(self, n_generations=1, n_episodes=1, render_episodes=0):
		"""
		render_episodes: number of episodes in between rendering. Zero (default) corresponds to not rendering at all.
		"""

		d = self.policy.params_dim
		eps = np.multivariate_normal(np.zeros(d), np.eye(d), size=n_episodes)

		def func(i_sample):
			self.policy.params = self.params + self.sigma*eps[i_sample]
			score1 = self.__perform_episode(render=(render_episodes>0) and (i_sample % render_episodes==0))

			self.policy.params = self.params - self.sigma*eps[i_sample]
			score2 = self.__perform_episode(render=(render_episodes>0) and (i_sample % render_episodes==0))
			return (score1 - score2)/(2*sigma) # CHECK IF THIS IS CORRECT!.. Estimate of gradient.

		scores = list(map(func, range(n_episodes)))
		self.params += self.alpha * np.inner(scores, eps)/n_episodes
			
	def scores(self, n_episodes, render_episodes=0):
		func = lambda i_episode: self.__perform_episode(train=False, render=(render_episodes>0) and (i_episode % render_episodes==0))
		return list(map(func, range(n_episodes)))
