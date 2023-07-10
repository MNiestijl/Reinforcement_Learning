from policy import NeuralPolicy

class ActorCritic(NeuralPolicy):

	""""
	Wrapper class that implements the general actor-critic framework
	action_space_size :: Int
	model: keras model with input shape (None, action_space_size), which takes the role of a actor, the policy.
	critic: Instance of Critic. Used to evaluate the actor.
	"""

	def __init__(self, state_dim, action_space_size, model, critic):
		self.critic = critic
		super().__init__(state_dim, action_space_size, model, get_targets=self.critic.get_targets)

	def train(self, experiences):
		self.critic.train(experiences)
		super().train(experiences)