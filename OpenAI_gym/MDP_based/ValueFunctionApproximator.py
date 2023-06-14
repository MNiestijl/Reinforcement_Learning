from Policy import Policy, NeuralPolicy
import utils as u
from MDP_based.ActorCritic import Q_Critic
import numpy as np
import numpy.random as rnd
import math as m
from copy import copy
from abc import ABCMeta, abstractmethod



class Neural_V_Learner():

	def __init__(self, model, discount, T=100):
		self.model = model
		self.target_model = copy(self.model)
		self.T = T
		self.reset_counter=1
		self.discount = discount

	def _get_bootsrap_value(self,s):
		s = np.array(s).reshape(1,len(s))
		return self.target_model.predict(s)[0,0]

	def _get_target(self, sold, a, snew, reward, done):
		return reward if done else reward + self.discount*max(self._get_bootsrap_values(snew))

	def evaluate(self, s):
		s = np.array(s).reshape(1,len(s))
		return self.model.predict(s)[0,0]

	def train_on_batch(self, experiences):
		states, actions, _, _ = zip(*experiences)
		n_samples = len(experiences)
		state_dim = len(states[0])
		targets = map(lambda x: self.get_target(*x), experiences)
		X = np.array(states).reshape(n_samples, state_dim)
		Y = np.zeros(n_samples, self.action_space_size)
		Y[:, actions] = targets
		self.model.train_on_batch(X,Y)

		if (self.reset_counter%self.T == 0):
			self.target_model = copy(self.model)
		self.reset_counter+=1

	def get_critic(self):
		return V_Critic(self)


class Neural_Q_Learner(NeuralPolicy):
	"""
	model: A neural network that maps State -> R^a (model(s) is a list of value functions for each action, so Q(s,a) = model(s)[a])
	T: Reset target_model every T timesteps
	""" 

	def __init__(self, action_space_size, model, discount, T=50):
		self.T = T
		self.discount = discount
		self.reset_counter=1
		super().__init__(action_space_size, model, self._get_target)
		self.target_model = copy(self.model)

	
	def _get_target(self, sold, a, snew, reward, done):
		"""
		Could use TD-lambda estimator instead, should be more robust.
		"""
		return reward if done else reward + self.discount*max(self._get_bootsrap_values(snew))

	def _get_bootsrap_values(self, s):
		s = np.array(s).reshape(1,len(s))
		return self.target_model.predict(s).flatten()

	def _evaluate_target_model(self,s,a):
		return self._get_bootsrap_values(s)[a]

	def get_values(self, s):
		s = np.array(s).reshape(1,len(s))
		return self.model.predict(s).flatten()

	def get_action(self, s):
		values = self.get_values(s)
		distribution= u.softmax(values)
		best = max(values)
		best_actions = list(filter(lambda a: values[a]==best, self.action_space))
		return rnd.choice(best_actions)
	
	def get_distribution(self, s):
		# Get a distribution using the softmax function over the approximated value function of all possible actions at state s.
		return u.softmax(self.get_values(s))

	def train_on_batch(self, experiences):
		super().train_on_batch(experiences)
		if (self.reset_counter%self.T == 0):
			self.target_model = copy(self.model)
		self.reset_counter+=1

	def get_critic(self):
		return Q_Critic(self)

