import numpy as np
import numpy.random as rnd
from Policy import Policy
from abc import ABCMeta, abstractmethod

class Critic(metaclass=ABCMeta):

	@abstractmethod
	def get_target(self, sold, a, snew, reward, done):
		pass

	@abstractmethod
	def update(self, sold, a, snew, reward, done):
		pass

class ActorCritic(Policy):

	""""
	Wrapper class that implements the actor-critic framework
	actor: Policy that implements update_by_target
	critic: Of instance of Critic.
	"""

	def __init__(self, actor, critic):
		self.actor = actor
		self.critic = critic
		super().__init__(actor.action_space_size)

	def get_distribution(self, s):
		return self.actor.get_distribution(s)

	def update(self, sold, a, snew, reward, done):
		self.critic.update(sold, a, snew, reward, done)
		target = self.critic.get_target(sold, a, snew, reward, done)
		self.actor.update_by_target(sold, a, target)

	def get_init_params(self):
		# Parameters are accessed trough the actor and critic, both of which have their own parameters.
		pass
		

class Q_Critic(Critic):

	def __init__(self, Q_learner):
		self.Q_learner = Q_learner

	def update(self, sold, a, snew, reward, done):
		self.Q_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, snew, reward, done):
		return self.Q_learner.evaluate(sold,a)


class V_Critic(Critic):

	def __init__(self, V_learner):
		self.V_learner = V_learner
	
	def update(self, sold, a, snew, reward, done):
		self.V_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, snew, reward, done):
		return self.V_learner.evaluate(sold, a)


class Advantage_Critic_V_Approx(Critic):

	def __init__(self, V_learner, discount):
		self.V_learner = V_learner
		self.discount = discount

	def update(self, sold, a, snew, reward, done):
		self.V_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, snew, reward, done):
		return reward + self.discount * self.V_learner.evaluate(snew) - self.V_learner.evaluate(sold)

class Advantage_Critic_Q_Approx(Critic):

	def __init__(self, Q_learner, discount):
		self.Q_learner = Q_learner
		self.discount = discount

	def update(self, sold, a, snew, reward, done):
		self.Q_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, snew, reward, done): 
		return self.Q_learner.evaluate(sold, a) - max(self.Q_learner.get_values(sold))


class Advantage_Critic_VQ_Approx(Critic):

	def __init__(self, V_learner, Q_learner, discount):
		self.V_learner = V_learner
		self.Q_learner = Q_learner
		self.discount = discount

	def update(self, sold, a, snew, reward, done):
		self.V_learner.update(sold, a, snew, reward, done)
		self.Q_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, snew, reward, done):
		return self.Q_learner.evaluate(sold, a) - self.V_learner.evaluate(sold)

class Advantage_Critic_VQ_Approx_Single_Network(Critic):

	def __init__(self, VQ_learner, discount):
		self.VQ_learner = VQ_learner
		self.discount = discount

	def update(self, sold, a, snew, reward, done):
		self.VQ_learner.update(sold, a, snew, reward, done)

	def get_target(self, sold, a, new, reward, done):
		Q = self.VQ_learner.get_Q_val(sold, a)
		V = self.VQ_learner.get_V_val(sold)
		return Q - V
