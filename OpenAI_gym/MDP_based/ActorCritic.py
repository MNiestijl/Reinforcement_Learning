import numpy.random as rnd
from Agent.AbstractAgent import AbstractAgent

class ActorCritic(AbstractAgent):

	""""
	Abstract class that implements policy gradient methods based on the policy gradient theorems.

	Input:
		- env: Gym environment
		- alpha: Learning rate.
		- actor: stochastic policy that implements the score function (gradient of log-likelihood)
		- critic: Value Function Approximator
	"""

	def __init__(self, env, actor, critic, observation=None, max_timesteps=100):
		super().__init__(env, observation, max_timesteps)
		self.actor = actor
		self.critic = critic
.
	def get_action(self, s):
		distribution = self.actor.evaluate(s) 
		return rnd.choice(actions, p=distribution)

	def update_params(self, sold, a, snew, reward):
		self.actor.params += self.actor.alpha*self.actor.score(sold, a)*self.critic.evaluate(sold, a)
		self.critic.update_params(sold, a, snew, reward)
		