import gym
import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import Input, Dense, Conv1D, Reshape, AveragePooling1D
from tensorflow.python.keras.constraints import maxnorm
from Policy import RandomPolicy, NeuralPolicy
from MDP_based.ActorCritic import Q_Critic, ActorCritic, Advantage_Critic_Q_Approx, Advantage_Critic_V_Approx
from MDP_based.MDPAgent import MDPAgent
from MDP_based.ValueFunctionApproximator import Neural_V_Learner, Neural_Q_Learner
import plot
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from itertools import permutations
from operator import mul
import math as m


"""
The stuff below is not very good! It would be way faster if there are multiple environments in which the actor obtains experience.
Since MDPAgent uses experience-replay, it is not compatible with the on-line actor critic methods. 
Also, the actor and critic should large share a feature network. 
"""

#  For compatibility with Tensorflow 1.
tf.compat.v1.disable_eager_execution()

def main():
	
	ENVIRONMENTS = {
	0: 'CartPole-v0',
	1: 'MountainCar-v0', 	# Hard due to reward function
	2: 'Ant-v2',
	3: 'Pendulum-v0',
	4: 'FrozenLake-v0',		# Easy, discrete state space
	5: 'Taxi-v2'			# Easy, discrete state space
	}
	discount=0.999
	max_timesteps = 1000
	environment = ENVIRONMENTS[0]
	env = gym.make(environment)
	action_space_size = env.action_space.n
	train_n_samples=100
	n_steps=5

	random_policy = RandomPolicy(action_space_size)
	random_agent = MDPAgent(env, random_policy, max_timesteps=max_timesteps)

	# act_funcs = get_act_funcs(env, n_funcs=100)
	# sarsa_approximator = NaiveApproximator(action_space, discount=0.6, act_funcs=act_funcs, alpha=0.4, method='SARSA')
	# naive_sarsa = MDPAgent( env, sarsa_approximator, eps=0.2, max_timesteps=max_timesteps)

	# act_funcs2 = get_act_funcs(env, n_funcs=100)
	# q_learning_approximator = NaiveApproximator(action_space, discount=0.6, act_funcs=act_funcs2, alpha=0.4)
	# naive_q_learning = MDPAgent( env, q_learning_approximator, max_timesteps=max_timesteps)

	
	# We assume the state space is of type Box here!
	assert type(env.observation_space) is gym.spaces.box.Box
	state_space_shape = env.observation_space.shape
	state_space_dim = state_space_shape[0]
	
	Q_model_1 = Sequential()
	Q_model_1.add(Dense(32, activation='relu', input_shape=state_space_shape))
	Q_model_1.add(Reshape((32,1)))
	Q_model_1.add(Conv1D(32, 3, padding='same', activation='relu'))
	Q_model_1.add(AveragePooling1D(pool_size=2))
	Q_model_1.add(Conv1D(64, 3, padding='same', activation='relu'))
	Q_model_1.add(AveragePooling1D(pool_size=2))
	Q_model_1.add(Conv1D(128, 3, padding='same', activation='relu'))
	Q_model_1.add(Reshape((128*8,)))
	Q_model_1.add(Dense(action_space_size, activation='linear'))
	Q_optimizer_1 = keras.optimizers.Adam(lr=1e-3, decay=1e-5)
	Q_model_1.compile(loss='mae', optimizer=Q_optimizer_1)

	Q_model_2 = Sequential()
	Q_model_2.add(Dense(128, activation='relu', input_shape=state_space_shape))
	Q_model_2.add(Dense(128, activation='relu'))
	Q_model_2.add(Dense(action_space_size, activation='linear'))
	Q_optimizer_2 = keras.optimizers.Adam(lr=1e-3, decay=1e-3)
	Q_model_2.compile(loss='mae', optimizer=Q_optimizer_2)
	
	Q_learner = Neural_Q_Learner(action_space_size, Q_model_1, discount=discount, T=100)
	Q_learner_2 = Neural_Q_Learner(action_space_size, Q_model_2, discount=discount, T=20)
	q_critic = Q_learner_2.get_critic()
	dqn_agent = MDPAgent(env, Q_learner, eps=0.3, eps_decay=1, max_timesteps=max_timesteps, train_n_samples=train_n_samples, n_steps=n_steps)

	V_model = Sequential()
	V_model.add(Dense(32, input_shape=state_space_shape, activation='relu'))
	V_model.add(Dense(32, activation='relu'))
	V_model.add(Dense(32, activation='relu'))
	V_model.add(Dense(1, activation='linear'))
	V_optimizer = keras.optimizers.Adam(lr=0.001, decay=1e-4)
	V_model.compile(loss='mse', optimizer=V_optimizer)

	V_learner = Neural_V_Learner(V_model, discount=discount, T=20)

	# # Create VQ model
	# VQ_input = Input(shape=state_space_shape)
	# x = Dense(32, activation='relu')(VQ_input)
	# x = Dense(32, activation='relu')(x)
	# q = Dense(8, activation='relu')(x)
	# Q_output = Dense(action_space_size, activation='linear', name='Q_output')(q)
	# V_output = Dense(1, activation='linear', name='V_output')(x)
	# VQ_model = Model(inputs=VQ_input, outputs=[ Q_output, V_output ])
	# VQ_optimizer = keras.optimizers.Adam(lr=1e-4, decay=1e-4)
	# VQ_model.compile(optimizer=VQ_optimizer, loss='mse')


	# VQ_learner = Neural_VQ_Learner(action_space_size, VQ_model, discount=discount, T=20)


	
	# advantage_critic_1 = Advantage_Critic_V_Approx(V_learner, discount)
	# # advantage_critic_2 = Advantage_Critic_VQ_Approx(V_learner, Q_learner, discount)
	# # advantage_critic_3 = Advantage_Critic_VQ_Approx_Single_Network(VQ_learner, discount)
	# advantage_critic_4 = Advantage_Critic_Q_Approx(Q_learner, discount)

	# actor_model_1 = Sequential()
	# actor_model_1.add(Dense(8, input_shape=state_space_shape, activation='relu'))
	# actor_model_1.add(Dense(8, activation='relu'))
	# actor_model_1.add(Dense(action_space_size, activation='softmax'))
	# actor_optimizer_1 = keras.optimizers.Adam(lr=1e-4, decay=1e-3)
	# actor_model_1.compile(loss='categorical_crossentropy', optimizer=actor_optimizer_1)

	# actor_model_2 = Sequential()
	# actor_model_2.add(Dense(8, input_shape=state_space_shape, activation='relu'))
	# actor_model_2.add(Dense(8, activation='relu'))
	# actor_model_2.add(Dense(8, activation='relu'))
	# #actor_model_2.add(Dense(32, activation='elu'))
	# actor_model_2.add(Dense(action_space_size, activation='softmax'))
	# actor_optimizer_2 = keras.optimizers.Adam(lr=1e-4, decay=1e-3)
	# actor_model_2.compile(loss='categorical_crossentropy', optimizer=actor_optimizer_2)

	# actor = NeuralPolicy(action_space_size, actor_model_1)
	
	# actor_critic = ActorCritic(actor, 	advantage_critic_4)
	# actor_critic_agent = MDPAgent(env, actor_critic, eps=0.01, eps_decay=0.999, max_timesteps=max_timesteps, train_n_samples=train_n_samples, n_steps=n_steps)


	agents = [
	#('Naive with eps', naive_q_learning_eps ),
	#('Naive', naive_q_learning),
	('Neural Q Learning', dqn_agent),
	# ('Actor Critic', actor_critic_agent),
	# ('Random Agent', random_agent),
	# ('Naive SARSA', naive_sarsa ),
	# ('Naive Q Learning', naive_q_learning),
	]
	
	#naive_q_learning.train(n_episodes=10, render_episodes=0)
	dqn_agent.scores(500) # Get some memory
	plot.plot_improvements(agents, n_episodes=2000, sample_size=1, train_size=1)

	if False:
		# for agent in agents:
		# 	agent.train(n_episodes=100)
		plot.plot_histograms(agents, n_episodes=10)
	
	plt.show()

	#actor_critic_agent.scores(1, render_episodes=1)


if __name__ == "__main__":
	main()
