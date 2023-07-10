import numpy as np
import random
from collections import deque


class MDPAgent():
	"""
	This makes use of "experience-replay" and therefore does not support online algorithms.
	"""

	def __init__(self, policy, get_env, memory_len=1e4):
		self.policy = policy
		self._get_env = get_env
		self.memory = deque(maxlen=memory_len)

	def _training_step(self, env, state, action):
		newstate, reward, terminated, truncated, info = env.step(action)
		done = terminated or truncated
		self.memory.append((state, action, newstate, reward, done))
		if done: 
			newstate = env.reset()[0]
		return newstate

	def gather_experience(self, envs, states, n_steps):
		for step in range(n_steps):
			X = np.stack(states)
			actions = self.policy.get_actions(X, training=True)
			states = [ self._training_step(env, state, action) for env, state, action in zip(envs, states, actions) ]

	def train(self, batch_size, n_epochs, n_steps, n_envs, repeat=1):
		"""
		Gather experience for n_steps in n_environments, and then train the policy n_epochs times,
		using samples from the current memory of size 'batch size'. This is repeated 
		We do allow batch_size > len(self.memory)!
		"""
		envs = [ self._get_env() for _ in range(n_envs)]
		states = [ env.reset()[0] for env in envs ]
		for _ in range(repeat):
			print('gathering experience')
			self.gather_experience(envs, states, n_steps)
			for _ in range(n_epochs):
				print('training an epoch')
				experiences = random.choices(self.memory, k=batch_size)
				self.policy.train(experiences)


	def score_episode(self, env, max_steps=-1):
		s = env.reset()[0]
		done = False
		r_tot = 0
		counter = 0
		while (not done and counter!=max_steps):
			action = self.policy.get_single_action(s)
			s, r, terminated, truncated, _ = env.step(action)
			done = terminated or truncated
			r_tot += r
			counter += 1
		env.close()
		return r_tot

	def score(self, n_episodes=1, max_steps=-1, render=False):
		"""
        Returns the total scores over the n episodes.
		"""
		env = self._get_env()
		return np.array([ self.score_episode(env, max_steps=max_steps, render=render) for _ in range(n_episodes) ])