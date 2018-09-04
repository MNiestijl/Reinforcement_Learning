import gym
import tensorflow as tf
from tensorflow import keras
from Policy import RandomPolicy
from MDP_based.ActorCritic import ActorCritic, Advantage_Critic_V_Approx, NeuralActor
from MDP_based.MDPAgent import MDPAgent
from MDP_based.ValueFunctionApproximator import NaiveApproximator, V_Approximator, Neural_Q_Learning
import plot
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from itertools import permutations
from operator import mul
import math as m

def _get_by_coords(array, coords):
	# Very strange way to perform array indexing... There should be a better way... right?
	if len(coords)==1:
		return array[coords[0]]
	else:
		first = coords[0]
		rest = coords[1:]
		return _get_by_coords(array[first], rest)

def get_act_funcs(env, n_funcs):
	# For now, it is assumed that env.observation_space is of type Box
	# Moreover, n_funcs should be an array matching the dimension of the observation space.

	# assert isinstanceof(env.observation_space, Box)
	# assert len(n_funcs)==env.observation_space.shape[0]

	state_space_type = type(env.observation_space)

	if state_space_type is gym.spaces.discrete.Discrete:
		n = env.observation_space.n
		act_func = lambda base_state: lambda input_state: int(base_state==input_state) # dirac-delta function at base_state
		return list(map(act_func, range(n)))
	elif state_space_type is gym.spaces.box.Box:

		BOX_BOUNDS = 100 # unbounded state spaces are not supported. value is arbitrary!

		high = np.array(list(map(lambda h: min(h, BOX_BOUNDS), env.observation_space.high)))
		low = np.array(list(map(lambda l: max(l, -BOX_BOUNDS), env.observation_space.low)))
		sizes = high - low
		dim = len(sizes)
		weights = list(map(lambda size : size/n_funcs, sizes))
		rbf = lambda mean: lambda x: m.exp(-np.inner((x-mean), (x-mean)/weights))/m.sqrt(2*m.pi*reduce(mul, weights,1))
		
		# Determine evenly spaced means in R^d... SHOULD BE DONE SIMPLER!
		n_base_points = m.floor(n_funcs**(1/float(dim)))
		get_base_vec = lambda d : np.linspace(low[d], high[d], n_base_points)
		base_vecs = list(map(get_base_vec, range(dim)))
		M = np.array(np.meshgrid(*base_vecs))
		dims = list(range(dim+1))
		M = np.transpose(M, dims[1:]+[dims[0]]) # Reshape is needed by implementation of _get_by_coords_

		inds_add_dim = lambda inds, d : [ ix+[j] for ix in inds for j in range(n_base_points)]
		inds = list(map(lambda x: [x], range(n_base_points)))
		for d in range(1,dim):
			inds = inds_add_dim(inds, d)
		means = [ _get_by_coords(M, ix) for ix in inds]
		return list(map(rbf, means))
	else:
		assert('type not supported')