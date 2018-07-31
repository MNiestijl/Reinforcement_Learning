from abc import ABCMeta, abstractmethod
from Policy import Policy
import numpy as np
import numpy.random as rnd
import math as m

class ValueFunctionApproximator(Policy, metaclass=ABCMeta):
	def __init__(self, action_space, discount):
		self.discount = discount
		super().__init__(action_space)

	# Evaluate the Q function at state, action.
	@abstractmethod
	def evaluate(self,s,a):
		pass

	def get_best_value(self, s):
		return max(map(lambda a: self.evaluate(s,a), self.action_space))

	def get_action(self, s):
		values = list(map(lambda a: (a, self.evaluate(s,a)), self.action_space))
		best = max(values, key=lambda x: x[1])[1]
		best_actions = list(map(lambda x: x[0], filter(lambda x: x[1]==best, values)))
		return rnd.choice(best_actions)

	def get_distribution(self, s):
		# Get stochastic policy using the softmax function over the approximated value function at a state.
		probalities = list(map(lambda a: m.exp(self.evaluate(s,a)), self.action_space))
		total = sum(probalities)
		return list(map(lambda p: p/total, probalities))


class NaiveApproximator(ValueFunctionApproximator):

	def __init__(self, action_space, discount, act_funcs, alpha=0.6,method='Q_learning'):
		self.act_funcs = act_funcs
		self.alpha = alpha
		self.method = method
		super().__init__(action_space, discount)

	def get_params_dim(self):
		return len(self.action_space) * len(self.act_funcs)

	def init_params(self):
		self.params = { a: np.zeros(len(self.act_funcs)) for a in self.action_space }

	def update_params(self, sold, a, snew, reward):
		if self.method=='Q_learning':
			TD = reward + self.discount*self.get_best_value(snew) - self.evaluate(sold,a)
		elif self.method=='SARSA':
			TD = reward + self.discount*self.evaluate(snew,a)- self.evaluate(sold,a)
		else:
			raise ValueError('The method {} is not supported'.format(method))
		self.params[a] += self.alpha*TD*self.eval_act_funcs(sold)

	def eval_act_funcs(self,s):
		evaluated = np.array([f(s) for f in self.act_funcs])
		normalized = evaluated/sum(evaluated)
		return normalized

	# Evaluate the Q function at state, action.
	def evaluate(self,s,a):
		return np.dot(self.params[a], self.eval_act_funcs(s))