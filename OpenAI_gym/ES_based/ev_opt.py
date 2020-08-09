import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to acces parent folder for imports.

import numpy as np
import math as m
import utils as u
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
		self.values.appendleft(x)
		if not self.saturated:
			self.length += 1
			self.current_value = 1/self.length * sum(self.values)
			self.saturated = (self.length == self.N)

	# Just for convenience.
	def get(self):
		return self.current_value
		
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
	def __init__(self, d_in, d_out, init_connections, init_params=None, act_fn=lambda x: max(0,x), 
		p_grow_outer=0.1, p_shrink_outer=0.12, p_grow_inner=0.1, p_shrink_inner=0.1, out_dimension_fixed=False, in_dimension_fixed=False):
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
		self.out_dimension_fixed = out_dimension_fixed
		self.in_dimension_fixed = in_dimension_fixed

		"""
		The following are parameters that could be adapted.
		"""
		self.covs = [ Layer._get_init_cov(d_w) for d_w in self.weight_dims ]


	def get_total_connections(self):
		return np.sum([len(ix) for ix in self.connections])

	@staticmethod
	def _get_cov_scale():
		return np.random.choice([2**x for x in range(-4,3)])

	@staticmethod
	def _get_init_cov(d_w):
		return Layer._get_cov_scale() * np.eye(d_w)

	def add_inputs(self, n):
		pass

	def add_outputs(self, n):
		pass
		# if self.out_dimension_fixed:
		# 	return
		# self.d_out = self.d_out + n
		# self.connections += [[]]*n
		# self.weights += [np.zeros(0)]*n
		# self.covs += [np.zeros((0,0))]
		# self.update()

	def remove_inputs(self, n):
		pass

	def remove_outputs(self, n):
		pass
		# if self.out_dimension_fixed or n>self.d_out:
		# 	return
		# inds = np.random.choice(range(self.d_out), size=n, replace=False)
		# for i in inds:
		# 	del self.connections[i]
		# 	del self.weights[i]
		# 	del self.covs[i]
		# self.update()

	def update(self):
		self.not_connected = [ list(set(range(self.d_in)) - set(ix)) for ix in self.connections]
		self.weight_dims = [ len(ix) for ix in self.connections ]

	def _evaluate(self, x, i):
		"""
		Determine the value of the i-th output dimension, with input x.
		"""
		ix = self.connections[i]
		w = self.weights[i]
		if not ix: # If list is empty
			return 0
		return self.act_fn(np.dot(w,x[ix]))


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
		"""
		Add for each output a new connection with an unconnected input, with some probability.
		"""
		if self.p_grow_outer>0 and bernoulli_trial(self.p_grow_outer):
			for i, _ in enumerate(self.connections):
				while self.p_grow_inner>0 and self.not_connected[i] and bernoulli_trial(self.p_grow_inner):
					new = np.random.choice(self.not_connected[i])
					self.connections[i].append(new)
					np.append(self.weights[i], 0) # Set the weight to the new connection to be zero.

					# make covariance matrix larger by extending a scalar diagonally, with zeros on the off-diagonals.
					d_w = self.weight_dims[i]
					zero_1 = np.zeros((d_w, d_w))
					zero_2 = np.zeros((1,1))
					scalar = np.array([[Layer._get_cov_scale()]])
					self.covs[i] = np.asarray(np.bmat([[self.covs[i], zero_2], [zero_1, scalar]]))
					self.update()
	def shrink(self):
		"""
		Remove for each output a connection with some probability.

		TODO: Currently, a connection to be deleted is chosen uniformly random. 
		Would be nice if this distribution is more well-informed, deleting the 'least useful' one more often.
		"""
		if self.p_shrink_outer>0 and bernoulli_trial(self.p_shrink_outer):
			for i, ix in enumerate(self.connections):
				while self.p_shrink_inner>0 and len(ix)>1 and bernoulli_trial(self.p_shrink_inner): # try to replace the >1 condition by ".. and ix and ...".
					i_rem = np.random.choice(range(n_connected))
					self.connections[i].remove(ix[i_rem])
					self.weights[i] = np.delete(self.weights[i], i_rem)
					self.covs[i] = np.delete(self.covs[i], i_rem, axis=0)
					self.covs[i] = np.delete(self.covs[i], i_rem, axis=1)
					self.update()


	def mutate(self):
		self.grow()
		self.shrink()


	def perturb_params(self):
		"""
		For now, we only perturb the coveriance matrix by multiples of the identity. Since it also starts this way, this means that
		it will always remain a multiple of the identity. 

		TODO!!
		This should be fixed, as it is unnecessary and hinders performance. More general perturbations should make use of cov = A^T A and perturb the matrix A instead, to ensure the matrix 
		is symmetric and positive definite. This should not be a major obstacle. 
		"""
		possible_scales = [2**x for x in range(-15,7)]

		for i, (d_w, cov) in enumerate(zip(self.weight_dims, self.covs)):
			if d_w==0:
				continue
			self.weights[i] += np.random.multivariate_normal(np.zeros(d_w), cov)
			self.covs += np.random.choice(possible_scales) * np.eye(d_w)




def get_new_layer(d_in, d_out, **kwargs):
	"""
	Each output is connected to a single input, which is chosen randomly.
	"""
	initial_connections = [ [np.random.choice(range(d_in))] for _ in range(d_out) ]
	return Layer(d_in, d_out, initial_connections, **kwargs)



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

	def get_total_connections(self):
		return sum([ layer.get_total_connections() for layer in self.layers ])

	def _perform_init_checks(self):
		# Input and output dimensions of the whole network.
		assert self.layers[0].d_in==self.d_in, "The input dimension does not match the specified dimension"
		assert self.layers[-1].d_out==self.d_out, "The output dimension does not match the specified dimension"
		assert 0<self.p_new_layer<1
		assert 0<self.p_new_node<1
		assert 0<self.p_remove_node<1

	def _n_inputs_of_layer_at_position(self, pos):
		pass


	def update(self):
		# If a layer has no connections, remove all subsequent layer and make a new layer with self.d_out outputs.
		i_unconnected = -1
		for i, layer in enumerate(self.layers):
			if not layer.connections:
				i_unconnected = i
		if i_unconnected != -1:
			self._insert_layer_at_position(i_unconnected)
			self.layers = self.layers[:i_unconnected]
			d_in = self._n_inputs_of_layer_at_position(i_unconnected)
			self.layers.append(get_new_layer(d_in. self.d_out))

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
			while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				first_layer.add_outputs(1)
				second_layer.add_inputs(1)
			while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
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


class NetworkSection1(NetworkSection):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _perform_init_checks(self):
		super()._perform_init_checks()
		
		# Input and output of subsequent layers
		for i, _ in range(self.n_layers-1):
			assert layers[i].d_out==layers[i+1].d_in, "The dimensions of the input and output of subsequent layers should match"


	def _n_inputs_of_layer_at_position(self, pos):
		return self.layers[pos-1].d_out if pos>0 else self.d_in

class NetworkSection2(NetworkSection):
	"""
	Uses all previous inputs + the last output as the input of a subsequent layer.
	"""
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)

	def _perform_init_checks(self):
		super()._perform_init_checks()

		# Input and output of subsequent layers
		for i, _ in range(self.n_layers-1):
			assert self.d_in + self.cum_out_dims[i]==layers[i+1].d_in, "The dimensions of the input and output of subsequent layers should match"


	def _n_inputs_of_layer_at_position(self, pos):
		return self.d_in + self.cum_out_dims[pos-1]if pos>0 else self.d_in

	def evaluate(self, inpt):
		output = inpt
		if not self.layers:
			return np.zeros(self.d_out)
		for layer in self.layers:
			output = np.concatenate((output, layer.evaluate(output)))
		return output
    

class Evolutionary_NN():
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

		How to do the third section? Maybe leave it out for now. Maybe the following contains something interesting?
		https://arxiv.org/pdf/1904.01554.pdf
	"""
	def __init__(self, section_1, section_2, d_out, p_new_node=0.01, p_remove_node=0.012):
		self.section_1 = section_1
		self.section_2 = section_2
		self.d_in = section_1.d_in
		self.d_out = d_out
		self.final_layer = get_new_layer(self.section_2.d_out, d_out, act_fn=lambda x: x, out_dimension_fixed=True)
		self.p_new_node = p_new_node
		self.p_remove_node = p_remove_node
		assert 0<self.p_new_node<1
		assert 0<self.p_remove_node<1

	def get_total_connections(self):
		return self.section_1.get_total_connections() + self.section_2.get_total_connections() + self.final_layer.get_total_connections

	def evaluate(self, x):
		"""
		Returns a probability distribution over the output dimensions
		"""
		x = self.section_1.evaluate(x)
		x = self.section_2.evaluate(x)
		x = self.final_layer.evaluate(x)
		return u.softmax(x)

	def sample_output(self, x):
		"""
		Samples from the distribution resulting from self.evaluate. Returns an integer in between (0, self.d_out)
		"""
		return np.random.choice(range(self.d_out), p=self.evaluate(x))

	def mutate(self):
		self.section_1.mutate()
		self.section_2.mutate()
		self.final_layer.mutate()

		# Make this into a function...
		while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				self.section_1.layers[-1].add_outputs(1)
				self.section_2.layers[0].add_inputs(1)
		while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
				self.section_1.layers[-1].remove_outputs(1)
				self.section_2.layers[0].remove_inputs(1)

		while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				self.section_2.layers[-1].add_outputs(1)
				self.final_layer[0].add_inputs(1)
		while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
				self.section_2.layers[-1].remove_outputs(1)
				self.final_layer[0].remove_inputs(1)


	def perturb_params(self):
		self.section_1.perturb_params()
		self.section_2.perturb_params()
		self.final_layer.perturb_params()



class EvolutionaryPolicyOptimizer():
	"""
	Optimize a class of policies in a certain gym environment.

	Currently assumes that the action space of tghe gym environment is just of the form set(range(n)) for some n,
	and that the output of initial_policy is precisely n.

	get_env - function that returns a gym environment
	initial_policy: instance of class Evolutionary_NN whose output dimension equals the dimension of the action space corresponding to the gym environment.
	"""
	def __init__(self, get_env, population_size, initial_policy, score_memory=100):
		self.population_size = population_size
		self.envs = [get_env() for _ in range(population_size)]
		self.states = [ env.reset() for env in self.envs ]
		self.policies = [initial_policy for _ in range(population_size)]
		self.rewards_avgs = [ RunningAverage(N=score_memory) for _ in range(population_size) ]
		# self.rewards_min = RunningMinimum(N=score_memory)


	def train(self, n_steps_total, n_steps_update, mode='average', callbacks=[]):
		"""
		possible modes: 'average' and 'safe', the former uses a moving average and the latter a moving minumum,
		to score various policies. 
		TODO: Implement 'safe' mode.
		"""
		for i in range(n_steps_total):
			for env, pol, state in zip(self.envs, self.policies, self.states):
				action = pol.sample_output(state)
				s_new, r, done, _  = env.step(action)
				r = r if not done else r-1 # This is done for compatibility with tasks of the form "stay alive as long as possible" that give the same award every step.
				self.rewards_avgs[i].update(r)
				self.states[i] = s_new if not done else env.reset()

			# Every ... steps, sample a new population from a distribution based on their average scores.
			if i % n_steps_update==0:
				scores = np.array([ avg.get() for avg in self.rewards_avgs ])
				scores -= np.min(scores)
				probs = scores/np.sum(scores)
				self.policies = np.random.choice(self.policies, size=self.population_size, p=probs, replace=True)

			for pol in self.policies:
				pol.mutate()
				pol.perturb_params()

			for callback in callbacks:
				callback(self)


	@staticmethod
	def _score_single_episode(env, pol, max_steps=-1):
		s = env.reset()
		done = False
		r_tot = 0
		counter = 0
		while not done and counter!=max_steps:
			action = pol.sample_output(s)
			s, r, done, _  = self.env.step(action)
			r_tot += r
			counter += 1
		return r_tot


	def score(self, n_episodes, max_steps=-1):
		"""
        Returns the total scores over the n episodes.
		"""
		env = self.get_env()
		avg_rewards = [ avg.get() for avg in self.rewards_avgs ]
		best_policy = self.policies[np.argmax(avg_rewards)]
		return [ EvolutionaryPolicyOptimizer._score_single_episode(env, pol, max_steps=max_steps) for _ in range(n_episodes) ]
