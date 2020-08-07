import numpy as np
import numpy.random as rnd
from copy import copy

class Policy():
	"""
	Policy interface.
	The policy may be either stochastic or deterministic. In the former case, implement the get_distribution method.
	In the latter case, implement the get_action method. At least one of these two must always be implemented.
	"""
	def __init__(self, action_space_size):
		self.action_space_size = action_space_size
		self.action_space = range(action_space_size)

	def update(self, sold, a, snew, reward, done):
		raise NotImplementedError("Online updating is not supported")

	def update_by_target(self, s, a, target):
		raise NotImplementedError("Updating by target is not supported")

	def get_distribution(self, s):
		raise NotImplementedError("The method 'get_distribution' is not implemented.")

	def get_stochastic_action(self, s):
		# Throws error if get_distribution is not implemented
		return rnd.choice(self.action_space, p=self.get_distribution(s))

	def get_action(self, s):
		try:
			return self.get_stochastic_action(s)
		except NotImplementedError:
			raise("At least one of the methods 'get_action' or 'get_distribution' must be implemented")

	# This method can be overridden.
	def train_on_batch(self, experiences):
		# Experience is of the form (sold, a, snew, reward)
		for experience in experiences:
			self.update(*experience)


class RandomPolicy(Policy):

	def __init__(self, action_space_size):
		super().__init__(action_space_size)

	def get_action(self, state):
		return rnd.choice(self.action_space)

	def update(self, sold, a, snew, reward, done):
		pass

class NeuralPolicy(Policy):
	"""
	model: A neural network that maps State -> Distribution over Actions
	target: A function mapping : sold, a, snew, reward, done -> R, that computes the target.
	""" 

	def __init__(self, action_space_size, model):
		self.action_space_size=action_space_size
		self.model=model

	def get_distribution(self, state):
		state = np.array(state).reshape(1,len(state))
		return self.model.predict(state).flatten()

	def update_by_target(self, s, a, target):
		X = np.array(s).reshape(1, len(s)) # Convert to 1 * dim_state_space matrix
		Y = np.zeros((1, self.action_space_size))
		Y[0, a] = target
		self.model.train_on_batch(X,Y)