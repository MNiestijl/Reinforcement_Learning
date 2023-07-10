import utils as u
import numpy as np
import scipy as sp
import numpy.random as rnd
import math as m
from abc import ABCMeta, abstractmethod
import tensorflow.keras as k
import tensorflow.math as tfmath
from copy import deepcopy

from policy import Policy, NeuralPolicy
from MDP_based.critic import Critic


@Critic.register
class Neural_V_Learner():

	def __init__(self, model, discount, T=100, mode = 'advantage'):
		self.T = T
		self.reset_counter=1
		self.discount = discount
		self.model = model
		self.target_model = k.models.clone_model(model)
		self.update_target_model()

	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	def _get_targets(self, experiences):
		"""
		Returns a np.array of shape (n_samples,)
		"""
		oldstates, _, newstates, rewards, dones = zip(*experiences)
		X_old = np.stack(oldstates)
		X_new = np.stack(newstates)
		R = np.stack(rewards)
		bootstrap_mask = np.where(dones, 0, 1) # recast dones as np.array of shape (n_sample), setting True to 0 and False to 1.
		BV_new = self.target_model.predict(X_new) # 'Bootstrap' value at X_new.
		
		match self.mode:
			case 'normal':
				return R + self.discount * np.dot(bootstrap_mask, BV_new)

			case 'advantage': 
				BV_old = self.target_model.predict(X_old)
				return R + self.discount * np.dot(bootstrap_mask, BV_new) - BV_old

	def get_values(self, states):
		return self.model.predict(states)

	def train(self, experiences):
		"""
		experience is a tuple of the form (sold, a, snew, reward, done).
		experiences :: [ experience ]
		"""
		states, actions, _, _ = zip(*experiences)
		n_samples = len(experiences)
		targets = self.get_targets(experiences)
		X = np.stack(states)
		Y = np.zeros(n_samples, self.action_space_size)
		Y[:, actions] = targets
		self.model.train_on_batch(X,Y)


		# TODO fix the part below!
		if ( self.reset_counter%self.T == 0 ):
			self.update_target_model()
		self.reset_counter+=1


@Critic.register
class Neural_Q_Learner(NeuralPolicy):
	"""
	model: A neural network that maps :: State -> R^a 
	(model(s) is a list of value functions for each action, so Q(s,a) = model(s)[a])
	T: Reset target_model every T experiences trained on
	mode : Possible values are 'normal' and 'advantage'.
	get_targets :: experiences -> np.array of shape (None, size_actions_space)
	""" 

	def __init__(self, state_dim, action_space_size, model, discount, T=None, mode='advantage', get_targets=None):
		self.T = T
		self.discount = discount
		self.reset_counter=1
		self.mode = mode
		if not get_targets:
			super().__init__(state_dim, action_space_size, model, self._get_targets)
		else: 
			super().__init__(state_dim, action_space_size, model, get_targets)
		if self.T:
			self.target_model = k.models.clone_model(model)
			self.update_target_model()
		self.alpha = 1000

	def update_target_model(self):
		if not self.T:
			return
		print('updating target model')
		self.target_model.set_weights(self.model.get_weights())
		self.reset_counter = 1

	@abstractmethod
	def get_bootstrap_values(Qvalues):
		"""
		From an array of Q values, of shape (n_samples, n_actions), determine the corresponding V-values,
		which is an array of shape (n_samples,)
		"""
		distributions = Neural_Q_Learner._Qvals_to_distributions(Qvalues)
		return np.sum(np.multiply(distributions, Qvalues), axis=1)


	def _get_targets(self, experiences):
		"""
		Returns a np.array of shape (n_samples, action_size)
		"""
		oldstates, actions, newstates, rewards, dones = zip(*experiences)
		oldstates = np.stack(oldstates)
		newstates = np.stack(newstates)
		R = np.stack(rewards)
		bootstrap_mask = np.where(dones, 0, 1) # recast dones as np.array of shape (n_sample,), setting True to 0 and False to 1.
		Qvals_new = self.target_model.predict(newstates)  
		BV_new = Neural_Q_Learner.get_bootstrap_values(Qvals_new) # 'Bootstrapped' V-values at newstates.

		# TODO: Implement TD(lambda) estimators.
		match self.mode:
			case 'normal':
				return R + self.discount * np.dot(bootstrap_mask, BV_new)

			case 'advantage': 
				Qvals_old = self.target_model.predict(oldstates)
				BV_old = Neural_Q_Learner.get_bootstrap_values(Qvals_old) # 'Bootstrapped' V-values at oldstates.
				return R + self.discount * np.multiply(bootstrap_mask, BV_new) - BV_old

	def get_values(self, states):
		"""
		A function :: n_samples × state_dim -> n_samples × n_actions

		The value of get_values(states)[sample_i, action_j] is the value of Q(state_i, action_j)
		"""
		return self.model.predict(states) 

	@staticmethod
	def _Qvals_to_distributions(Qvalues):
		"""
		Transform the Q values, an array of shape (n_samples, n_actions), 
		into an array of probability distributions over the actions, of the same shape. 
		"""
		return sp.special.softmax(Qvalues, axis = -1)

	def get_distributions(self, states):
		"""
		Apply softmax over the Q values Q(s,a) in state s over all actions a.
		The result is of shape (n_samples, n_actions).
		"""
		Qvalues = self.get_values(states)  # of shape (n_samples, n_actions)
		return Neural_Q_Learner._Qvals_to_distributions(Qvalues)

	def train(self, experiences):
		chunks = [experiences[i:i + self.T] for i in range(0, len(experiences), self.T)]
		for chunk in chunks:
			if self.reset_counter >= self.T:
				self.update_target_model()
			super().train(chunk)
			self.reset_counter += len(chunk)
			print('training a chunk of {} experiences'.format(len(chunk)))
		


@Critic.register
class Neural_VQ_learner(Neural_Q_Learner):
	def __init__(self, action_space_size, get_Q_model, get_V_model, discount, T=500, mode='advantage'):
		self.V = Neural_V_Learner(get_V_model, discount, T=T, mode=mode)
		super().init(action_space_size, get_Q_model, discount, T=T, mode=mode)

	def get_targets(self, experienes):
		oldstates, actions, _, _, _ = zip(*experiences)
		oldstates = np.stack(oldstates)
		newstates = np.stack(newstates)
		Q_values =  np.choose(actions, self.get_values(oldstates).T)
		V_values = self.V.get_values(oldstates)
		return Q_values - V_values

	def train(self, experiences):
		self.V.train(experiences)
		super().train(experiences)