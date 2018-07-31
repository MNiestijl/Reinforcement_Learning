import numpy as np
import numpy.random as rnd

class MDPAgent():
	#TODO: Implement 'experience replay' to make algorithm more efficient and avoid bad feedback loops
	# TODO: Make it such that agent uses distributions during training if possible but not during scoring.

	def __init__(self, env, policy, observation=None, max_timesteps=100, eps=0):
		self.policy = policy
		self.env = env
		self.observation = observation if observation else env.reset()
		self.max_timesteps = max_timesteps
		self.eps = eps

	def __perform_episode(self,train=True, render=False):
		print("New Episode")
		total_reward = 0
		self.observation = self.env.reset()
		for timestep in range(self.max_timesteps):
			if render:
				self.env.render()

			# Get action
			if train and (0<self.eps<1) and rnd.choice([True, False],size=1, p=[self.eps,1-self.eps]):
				action = self.env.action_space.sample()
			elif train:
				try: # During training, use stochastic action if possible
					action = self.policy.get_stochastic_action(self.observation)
				except:
					action = self.policy.get_action(self.observation)
			else:
				action = self.policy.get_action(self.observation)
			
			# Update everything
			observation, reward, done, info = self.env.step(action)
			if train:
				self.policy.update_params(self.observation, action, observation, reward)
			self.observation = observation
			total_reward += reward
			if done:
				break
		return total_reward

	def train(self, n_episodes=1, render_episodes=0):
		"""
		render_episodes: number of episodes in between rendering. Zero (default) corresponds to not rendering at all.
		"""
		func = lambda i_episode: self.__perform_episode(render=(render_episodes>0) and (i_episode % render_episodes==0))
		return list(map(func, range(n_episodes)))
			
	def scores(self, n_episodes, render_episodes=0):
		func = lambda i_episode: self.__perform_episode(train=False, render=(render_episodes>0) and (i_episode % render_episodes==0))
		return list(map(func, range(n_episodes)))
