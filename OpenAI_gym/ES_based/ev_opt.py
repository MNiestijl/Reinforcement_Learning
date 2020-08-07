import numpy as np
import math as m
from collections import Counter, deque
from abc import ABCMeta, abstractmethod


class EvolutionaryOptimizer():
	"""
	obj_fun: Real-valued function to be optimized.
	obj_fun :: Array(N × input_dim) -> Array(N)
	
	initial_inputs: Array(N × input_dim).

	covariances :: [ Array(input_dim × input_dim) ] - Covariances of normal distributions 
	cov_probs - List of probabilities, should have the same length as covariances and sum to one.
	"""
	def __init__(self, obj_fun, initial_inputs, covariances, cov_probs=None):
		cov_probs = [ 1/len(covariances) ] * len(covariances) if cov_probs is None else cov_probs
		assert len(cov_probs)==len(covariances), "cov_probs and covariances should have the same length"
		assert m.isclose(sum(cov_probs),1) and all(p >= 0 for p in cov_probs), "cov_probs should define a probability distribution"

		self.inputs = initial_inputs
		self.N, self.input_dim = initial_inputs.shape
		self.evaluate = lambda : obj_fun(self.inputs)
		self.covariances = covariances
		self.cov_probs = cov_probs
		self.zeros = np.zeros(self.input_dim) # to avoid creating this list again for each call of "step".

	def step(self):
		values = self.evaluate()
		values -= np.min(values)
		if any(v > 0 for v in values):
			distribution = values/np.sum(values)
		else:
			distribution = np.array([1/len(values)] * len(values))
		children_ix = np.random.choice(self.N, size=self.N, p=distribution)
		covariances_ix = np.random.choice(len(self.covariances), size=self.N, p=self.cov_probs)
		cov_occurrances = Counter(covariances_ix)
		self.inputs = self.inputs[children_ix, :]
		epsilons = [ np.random.multivariate_normal(self.zeros, self.covariances[i], size=N) for i, N in cov_occurrances.items() ]
		epsilons = np.concatenate(epsilons)
		self.inputs += epsilons



class RunningAverage():
	def __init__(self, N):
		self.N = N
		self.reset()

	def update(self, x):
		if self.saturated:
			oldest = self.values.pop()
			self.current_value += 1/N * (x - oldest)
		self.values.insert(0, x)
		if not self.saturated:
			self.length += 1
			self.current_value = 1/self.length * sum(self.values)
			self.saturated = (self.length == self.N)
		
	def reset(self):
		self.values = deque(maxlen=self.N)
		self.current_value = 0
		self.saturated = False
		self.length = 0		

class RunningMinimum():
	""" TODO """
	pass


def bernoulli_trial(p):
	return np.random.binomial(1, p, 1)


class Layer():
	def __init__(self, d_in, d_out, init_connections, init_params=None, act_fn=lambda x: np.max(0,x), 
		p_grow_outer=0.01, p_shrink_outer=0.02, p_grow_inner=0.1, p_shrink_inner=0.1):
		"""
		d_in and d_out are input/output dimensions.
		init_connections is supposed to be a list of length d_out containing a lists of input indices for each output.
		"""
		self.d_in = d_in
		self.d_out = d_out
		self.connections = init_connections
		self.act_fn = act_fn
		self.update()

		self.weights = [ np.zeros(d_w) for d_w in self.weight_dims ]
		self.p_grow_outer = p_grow_outer
		self.p_grow_inner = p_grow_inner
		self.p_shrink_outer = p_shrink_outer
		self.p_shrink_inner = p_shrink_inner

		"""
		The following are parameters that could be adapted.
		"""

		scale = np.random.choice([2**x for x in range(-4,3)])
		self.covs = [ np.eye(d_w) for d_w in weight_dims ]
		self.covs_eps = [2**x for x in range(-15,7)]

	def add_inputs(self, n_new):
		pass #TODO

	def add_outputs(self, n_new):
		pass #TODO

	def remove_inputs(self, n_new):
		pass #TODO

	def remove_outputs(self, n_new):
		pass #TODO

	def update(self):
		self.not_connected = [ list(set(range(0, d_in)) - set(ix)) for ix in self.connections]
		self.weight_dims = [ len(ix) for ix in self.connections ]

	def _evaluate(self, x, i):
		"""
		Determine the value of the i-th output dimension, with input x.
		"""
		ix = self.connections[i]
		w = self.weights[i]
		if not ix: # If list is empty
			return 0
		return self.act_fn(np.dot(w,ix))


	def evaluate(self, x):
		"""
		Return zero if there are no connections

		This can maybe be done more efficiently by concatenating the input slices and using "reduceat" appropriately.
		Also using binary masks, resulting in a (d_in × d_out)-sized **sparse** array could be useful.
 		"""
 		if not self.connections: # if list is empty
 			return np.zeros()
 		return np.array([ self._evaluate(x, i) for i in range(len(self.connections)) ])

	def grow(self):
		if self.p_grow_outer > 0 and bernoulli_trial(self.p_grow_outer):
			for i, _ in enumerate(self.connections):
				if self.p_grow_inner > 0 and bernoulli_trial(self.p_grow_inner):
					new = np.random.choice(self.not_connected[i])
					self.connections[i].append(new)
					np.append(self.weights[i], 0) # Set the weight to the new connection to be zero.
	def shrink(self):
		pass #TODO


	def mutate(self):
		"""
		Can add an index or remove and index, or both.
		"""

		self.grow()
		self.shrink()
		self.update()

	def perturb_params(self):
		epsilons = [ np.random.multivariate_normal(self.zeros(d_w), cov) for d_w, cov in zip(self.weight_dims, self.covariances) ]
		self.weights = [ w + e for w,e in zip(self.weights, epsilons)]
		cov_eps = [ np.random.choice(self.cov_eps) for _ in self.weight_dims ]
		self.covs = [ cov + e for cov, e in zip(self.covs, cov_eps) ]


def get_new_layer(d_in, d_out):
	"""
	Each output is connected to a single input, which is chosen randomly.
	"""
		initial_connections = [ np.random.choice(range(d_in)) for _ in range(d_out) ]
		return Layer(d_in, d_out, initial_connections)



class NetworkSection():
	def __init__(self, layers, d_in, d_out, p_new_layer=0.005, p_new_node=0.05, p_remove_node=0.06):
		self.layers = layers
		self.d_in = d_in
		self.d_out = d_out
		self.p_new_layer = p_new_layer
		self.p_new_node = p_new_node
		self.p_remove_node = p_remove_node
		self.update()
		self._perform_init_checks()

	def _perform_init_checks(self):
		# Input and output dimensions of the whole network.
		assert layers[0].d_in==self.d_in, "The input dimension does not match the specified dimension"
		assert layers[-1].d_out==self.d_out, "The output dimension does not match the specified dimension"

	def _n_inputs_of_layer_at_position(self, pos):
		pass


	def update(self):
		# If a layer has no connections, remove all subsequent layer and make a new layer with self.d_out outputs.
		i_unconnected = -1
		for i, layer in enumerate(self.layers):
			if not layer.connections: # check if list is empty
				i_unconnected = i
		if i_unconnected != -1:
			self._insert_layer_at_position(i_unconnected)
			self.layers = self.layers[:i_unconnected]
			d_in = self._n_inputs_of_layer_at_position(i_unconnected)
			#d_in = self.layers[-1].d_out
			d_out = self.d_out
			self.layers.append(get_new_layer(d_in. d_out))

		self.n_layers = len(self.layers)
		self.cum_out_dims = np.cumsum([ layer.d_out for layer in self.layers])


	def evaluate(self, inpt):
		output = inpt
		if not self.layers:
			return np.zeros(self.d_out)
		for layer in self.layers:
			output = layer.evaluate(output)
		return output


	def mutate(self):

		# Mutate layers
		for layer in self.layers:
			layer.mutate()

		# Add and remove nodes in between layers
		for i in range(self.n_layers-1):
			first_layer, second_layer = self.layers[i], self.layers[i+1]
			if self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				first_layer.add_outputs(1)
				second_layer.add_inputs(1)
			if self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
				first_layer.remove_outputs(1)
				second_layer.remove_inputs(1)

		# Add new layer
		if self.p_new_layer>0 and bernoulli_trial(self.p_new_layer):
			pos = np.random.choice(range(len(self.layers)-1))
			d_in = self._n_inputs_of_layer_at_position(pos)
			d_out = self.layers[pos+1].d_in
			self.layers.insert(pos, get_new_layer(d_in, d_out))

		self.update()


	def perturb_params(self):
		for layer in self.layers:
			layer.perturb_params()


class NetworkSection1(NetworkSection)
	def __init__(self, *args, **kwargs)
		super().__init__(*args, **kwargs)

	def _perform_init_checks(self):
		super()._perform_init_checks()
		
		# Input and output of subsequent layers
		for each i, _ in range(self.n_layers-1):
			assert layers[i].d_out==layers[i+1].d_in, "The dimensions of the input and output of subsequent layers should match"


	def _n_inputs_of_layer_at_position(self, pos):
		return self.layers[pos-1].d_out if pos>0 else self.d_in

class NetworkSection2(NetworkSection):
	"""
	Uses all previous inputs + the last output as the input of a subsequent layer.
	"""
	def __init__(self, *args, **kwargs)
		super().__init__(*args, **kwargs)
		self.input_dims = self.

	def _perform_init_checks(self):
		super()._perform_init_checks()

		# Input and output of subsequent layers
		for each i, _ in range(self.n_layers-1):
			assert self.d_in + self.cum_out_dims[i]==layers[i+1].d_in, "The dimensions of the input and output of subsequent layers should match"


	def _n_inputs_of_layer_at_position(self, pos):
		return self.layers[pos-1].d_out if pos>0 else self.d_in

	def evaluate(self, inpt):
		output = inpt
		if not self.layers:
			return np.zeros(self.d_out)
		for layer in self.layers:
			output = np.concatenate((output, layer.evaluate(output)))
		return output



class evolutionary_NN():
	"""
	Should contain four sections:
	1: layers that do not remember previous outputs
	2: layers that DO remember previous outputs (from this second part)
	3: identity gates, On/off gates, Logical gates; including <=, >=, NOT, AND, OR, XOR. 
		(Some of these are also possible to form in sections 1 and 2)
	4: Final probability distribution/softmax layer

	First and second sections include ReLu's at the end of each layer.

	Maybe make a Layer class for each of the four sections, that implements the same methods.
		For example: If a layer has A inputs and B outputs, then have B lists of input indices (of varying lengths.
		For each output b, take the corresponding input slice and multiply it by the corresponding weight vector. 
		Then sum and apply ReLu. 

		How to do the third section? Maybe leave it out for now.
	"""
	def __init__(self, section_1, section_2):
		pass

	def evaluate(self, state):
		pass

	def mutate(self):
		"""
		For each sections, define possibly mutations. 
		Both growing and decreasing the network should be possible. 
		"""
		pass

	def perturb_params(self):
		pass

def pair_ENN(network_1, network_2):
	pass


class EvolutionaryPolicyOptimizer():
	"""
	Optimize a class of policies in a certain gym environment.
	"""
	def __init__(self, get_env, population_size, initial_policy, p_mutate):
		assert 0 <= p_mutate <= 1
		self.population_size = population_size
		self.envs = [get_env() for _ in range(population_size)]
		self.states = [ env.reset() for env in self.envs ]
		self.initial_policy = initial_policy
		self.policies = [initial_policy for _ in range(population_size)]
		self.rewards_avg = RunningAverage(N=100)
		self.rewards_min = RunningMinimum(N=100)
		self.step_fns = [ lambda : env.step(pol.get_action(s)) for env, s, pol in zip(self.envs, self.states, self.policies) ]

	def train(self, n_repeats, n_steps=1, mode='average'):
		"""
		possible modes: 'average' and 'safe', the former uses a moving average and the latter a moving minumum,
		to score various policies.

		TODO:
		- run step functions ----> yields rewards. Use these to update running averages and running minima
		- for each policy: 
			---PAIRING---
			- with probability p_pair, sign policy up to be paired. Remove one if total signed up policies is odd.
			- Devide into pairs. 
			Combine their parameters as follows: 
			- mode 'best': Copy the network of one of the two, with a probability based on the scores.
			- mode 'average': Copy the network of one of the two, with a probability based on the scores. 
			Copy its parameters on connections that are not shared
			Average their parameters for shared connections with weights.

			---Mutating---
			with probability p_mutate, apply policy.mutate(). This is supposed to alter the skeleton of the network,
			and hence also the shapes of the parameters

			---Perturbation---
			with probability p_perturb, add gaussian noise to the parameters, with mean 0 and a covariance matrix that is part of the parameters!

		"""