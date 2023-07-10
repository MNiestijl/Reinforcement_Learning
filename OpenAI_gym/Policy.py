import numpy as np
import numpy.random as rnd
from copy import copy
import utils as u

class Policy():
	"""
	Policy interface.
	The policy may be either stochastic or deterministic. In the former case, implement the get_distributions method.
	In the latter case, implement the get_actions method. At least one of these two must always be implemented.
	"""
	def __init__(self, state_dim, action_space_size):
		self.state_dim = state_dim
		self.action_space_size = action_space_size
		self.action_space = range(action_space_size)

	def get_distributions(self, states):
		raise NotImplementedError("The method 'get_distributions' is not implemented.")

	def get_stochastic_actions(self, states):
		"""
		and sample the probability distribution.
		The result is of shape (n_samples,).

		Throws error if get_distribution is not implemented
		"""
		return u.sample_indices_from_2D_array(self.get_distributions(states), axis=1)

	def get_actions(self, states):
		"""
		A function of the form: state_dim -> np.array of shape (n_samples,)
		"""
		try:
			return self.get_stochastic_actions(states)
		except NotImplementedError:
			raise("At least one of the methods 'get_actions' or 'get_distributions' must be implemented")

	def get_single_action(self, state):
		return self.get_actions(np.reshape(state, (1, self.state_dim)))[0]

	# 
	def train(self, experiences):
		"""
		This method should be overridden in subclasses.
		
		Experience is a tuple of the form (sold, action, snew, reward, done)
		Experiences is a list [ experience ]
		"""
		pass



class RandomPolicy(Policy):

	def __init__(self, action_space_size):
		super().__init__(action_space_size)

	def get_action(self, states):
		return rnd.choice(self.action_space)


class NeuralPolicy(Policy):
	"""
	model: A neural network that maps :: np.array(state_dim,) -> Distribution over Actions
	get_targets: Function :: experiences -> array,
	where expereriences :: [ experience],
	with experience :: (sold, a, snew, rew, done) 
	""" 

	def __init__(self, state_dim, action_space_size, model, get_targets):
		self.model=model
		self.get_targets = get_targets
		super().__init__(state_dim, action_space_size)

	def get_distributions(self, states):
		"""
		states should be an array of shape: n_samples × state_dim
		"""
		return self.model.predict(states)


	def train(self, experiences):
		"""
		experience is a tuple (sold, a, snew, reward, done)

		"""
		states, actions, _, _, _ = zip(*experiences)
		n_samples = len(experiences)
		targets = self.get_targets(experiences)
		X = np.stack(states)
		Y = np.zeros((n_samples, self.action_space_size))
		Y[:, actions] = targets
		self.model.train_on_batch(X,Y)