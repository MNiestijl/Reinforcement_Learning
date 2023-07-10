import abc

class Critic(abc.ABC):

	@abc.abstractmethod
	def get_targets(self, experiences):
		pass

	@abc.abstractmethod
	def train(self, experiences):
		pass