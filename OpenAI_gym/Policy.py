import numpy.random as rnd
from abc import ABCMeta, abstractmethod
from copy import copy

class Policy(metaclass=ABCMeta):
	"""
	The policy may be either stochastic or deterministic. In the former case, implement the get_distribution method.
	In the latter case, implement the get_action method. At least one of these two must always be implemented.

	For MDP-based algorithms, the method 'update_params' must be implemented.

	"""

	def __init__(self, action_space):
		self.action_space = action_space
		self.init_params()
		self.params_dim = self.get_params_dim()

	@abstractmethod 
	def get_params_dim(self):
		pass

	@abstractmethod
	def init_params(self):
		pass

	def get_params(self):
		return self.params.copy()

	def update_params(self, old_observation, action, new_observation, reward):
		raise NotImplementedError("Online updating is not supported")

	def get_distribution(self, s):
		raise NotImplementedError("The method 'get_distribution' is not implemented.")

	def get_action(self, s):
		try:
			return self.get_stochastic_action(s)
		except NotImplementedError:
			raise("At least one of the methods 'get_action' or 'get_distribution' must be implemented")

	def get_stochastic_action(self, s):
		# Throws error if get_distribution is not implemented
		return rnd.choice(self.action_space, p=self.get_distribution(s))


class RandomPolicy(Policy):

	def __init__(self, action_space):
		super().__init__(action_space)

	def get_params_dim(self):
		return 0

	def get_action(self, observation):
		return rnd.choice(self.action_space)

	def update_params(self, old_observation, action, new_observation, reward):
		pass

	def init_params(self):
		pass