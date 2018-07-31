import numpy.random as rnd
from abc import ABCMeta, abstractmethod

class Policy(metaclass=ABCMeta):
	"""
	The policy may be either stochastic or deterministic. In the former case, implement the get_distribution method.
	In the latter case, implement the get_action method. At least one of these two must always be implemented.

	For MDP-based algorithms, the method 'update_params_online' must be implemented.
	
	"""

	def __init__(self, action_space):
		self.action_space = action_space
		self.params = self.get_init_params()
		self.params_dim = len(self.params)
		self.stochastic = False # Whether or not to sample actions according to the probability distribution.

	@abstractmethod
	def init_params(self):
		pass

	def get_params(self):
		raise NotImplementedError("The method 'get_params' is not implemented.")

	def update_params_online(self, old_observation, action, new_observation, reward):
		except NotImplementedError:
			raise("Online updating is not supported")

	def get_distribution(self, s):
		raise NotImplementedError("The method 'get_distribution' is not implemented.")

	def get_action(self, s):
		try:
			self.get_stochastic_action(s)
			self.stochastic = True
		except NotImplementedError:
			raise("At least one of the methods 'get_action' or 'get_distribution' must be implemented")

	def get_stochastic_action(self, s):
		return rnd.choice(self.action_space, p=self.get_distribution(s))


class RandomPolicy(Policy):

	def __init__(self, action_space):
		super().__init__(action_space)

	def get_action(self, observation):
		return rnd.choice(self.action_space)

	def update_params_online(self, old_observation, action, new_observation, reward):
		pass

	def init_params(self):
		pass