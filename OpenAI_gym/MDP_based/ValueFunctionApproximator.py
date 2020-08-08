from Policy import Policy, NeuralPolicy
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
		Could TD-lambda estimator instead, should be more robuust.
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
		best = max(values)
		best_actions = list(filter(lambda a: values[a]==best, self.action_space))
		return rnd.choice(best_actions)
	
	def get_distribution(self, s):
		# Get a distribution using the softmax function over the approximated value function of all possible actions at state s.
		vals = self.get_values(s)
		max_val = max(vals)
		xs = list(map(lambda a: m.exp(vals[a]-max_val), self.action_space))
		total = sum(xs)
		return list(map(lambda x: x/total, xs))

	def train_on_batch(self, experiences):
		super().train_on_batch(experiences)
		if (self.reset_counter%self.T == 0):
			self.target_model = copy(self.model)
		self.reset_counter+=1

	# def update_by_target(self, s, a, target):
	# 	if (self.reset_counter%self.T == 0):
	# 		self.target_model = copy(self.model)
	# 	X = np.array(s).reshape(1, len(s)) # Convert to 1 * dim_state_space matrix (batch_size * dim_s)
	# 	Y = np.zeros((1, self.action_space_size))
	# 	Y[0, a] = target
	# 	self.model.train_on_batch(X,Y)
	# 	self.reset_counter+=1

	# def update(self, sold, a, snew, reward, done):
	# 	target = reward if done else reward + self.discount*max(self._get_bootsrap_values(snew))
	# 	self.update_by_target(sold, a, target)

	def get_critic(self):
		return Q_Critic(self)





# class Neural_VQ_Learner(NeuralPolicy):
# 	"""
# 	Learn both V and Q simultaneously.
# 	model: A neural network that maps State -> R^(a+1) 
# 	convention: Q(s,a) = model(s)[a] and V(s) = model(s)[-1]

# 	T: Reset target_model every T timesteps
# 	""" 

# 	def __init__(self, action_space_size, model, discount, T=50):
# 		self.model = model
# 		self.target_model = copy(self.model)
# 		self.T = T
# 		self.reset_counter=1
# 		self.action_space_size = action_space_size
# 		self.discount=discount

# 	def eval_target_model(self, s):
# 		s = np.array(s).reshape(1, len(s))
# 		return self.target_model.predict(np.array(s))

# 	def _get_target_Q_values(self, s):
# 		return self.eval_target_model(s)[0].flatten()

# 	def get_target_Q_val(self,s,a):
# 		return self._get_target_Q_values(s)[a]

# 	def get_target_V_val(self, s):
# 		return self.eval_target_model(s)[1][0,0]

# 	def eval_model(self, s):
# 		s = np.array(s).reshape(1, len(s))
# 		return self.model.predict(s)

# 	def get_Q_values(self, s):
# 		return self.eval_model(s)[0].flatten()

# 	def get_Q_val(self, s, a):
# 		return self.get_Q_values(s)[a]		

# 	def get_V_val(self, s):
# 		return self.eval_model(s)[1][0,0]

# 	def update(self, sold, a, snew, reward, done):
# 		V_target = reward if done else reward + self.discount * self.get_target_V_val(snew)
# 		Q_target = reward if done else reward + self.discount*max(self._get_target_Q_values(snew))
# 		if (self.reset_counter%self.T == 0):
# 			self.target_model = copy(self.model)
# 		X = np.array(sold).reshape(1, len(sold)) # Convert to 1 * dim_state_space matrix
# 		Y_Q = np.zeros((1, self.action_space_size))
# 		Y_Q[0, a] = Q_target
# 		Y_V = np.array([[V_target]])
# 		self.model.train_on_batch(X,[Y_Q, Y_V])
# 		self.reset_counter+=1

# 	def get_critic(self):
# 		return Q_Critic(self)








# class NaiveApproximator(Policy):
# 	# NOTICE: Currently, the parameters are NOT stored as vectors, thus it is not (yet) compatible with ES-based methods.
# 	# This should be fixed later.

# 	# TODO: Discount parameter should not really belong to this class; but rather the definition of the MDP itself... 

# 	def __init__(self, action_space, discount, act_funcs, alpha=0.6,method='Q_learning'):
# 		self.discount = discount
# 		self.act_funcs = act_funcs
# 		self.alpha = alpha
# 		self.method = method
# 		self.params = { a: np.zeros(len(self.act_funcs)) for a in self.action_space }
# 		super().__init__(action_space, discount)

# 	def update(self, sold, a, snew, reward):
# 		if self.method=='Q_learning':
# 			TD = reward + self.discount*self.get_best_value(snew) - self.evaluate(sold,a)
# 		elif self.method=='SARSA':
# 			TD = reward + self.discount*self.evaluate(snew,a)- self.evaluate(sold,a)
# 		else:
# 			raise ValueError('The method {} is not supported'.format(method))
# 		self.params[a] += self.alpha*TD*self.eval_act_funcs(sold)

# 	def eval_act_funcs(self,s):
# 		evaluated = np.array([f(s) for f in self.act_funcs])
# 		normalized = evaluated/sum(evaluated)
# 		return normalized

# 	# Evaluate the Q function at state, action.
# 	def evaluate(self,s,a):
# 		return np.dot(self.params[a], self.eval_act_funcs(s))
