import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to acces parent folder for imports.

import numpy as np
import math as m
import utils as u
from collections import Counter, deque
from abc import ABCMeta, abstractmethod
from copy import copy, deepcopy

import pdb

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

	def reset(self):
		self.values = deque(maxlen=self.N)
		self.current_value = 0
		self.saturated = False
		self.length = 0	

	def update(self, x):
		if self.saturated:
			oldest = self.values.pop()
			self.current_value += 1/self.N * (x - oldest)
		self.values.appendleft(x)
		if not self.saturated:
			self.length += 1
			self.current_value = 1/self.length * sum(self.values)
			self.saturated = (self.length == self.N)

	# Just for convenience.
	def get(self):
		return self.current_value
	

class RunningMinimum():
	""" TODO """
	pass


def bernoulli_trial(p):
	return np.random.binomial(1, p, 1)


"""
Implement parallel CPU usage using the multiprocessing package.
"""


"""
If the algorithms are ran in parallel, and also compete in speed somehow, faster networks (i.e., smaller ones), will automatically be preferred.
Would be nice to implement this idea somehow, at some point.
"""


"""
TODO: 
Make tree-like structure for the policies below, such that a policy can easily be "engineered" out of layers, sections, memories etc.
in a keras-like manner, that supports having multiple sources of inputs. Should be able to add policies or layers 'at the end' of some existing policy.
(In the tree-structure, the top-node corresponds to the output, all the leaves to the inputs, nodes to layers, sections to 'subtrees'.)
"""


"""
TODO!!
Make 2D convolution layer/section. Would be great if the system does not all parts of the input equally...
Perhaps a smaller subsystem selects N pixels to-be-taken-as-input, which are ordered based on the distance of their indices.
In any case, the idea is to down-size the input in such a way that minimal information is lost.
Perhaps there is some other way to do this.
"""

"""
TODO: Make a memory section, that is; at some stage in the network, the output at that state fed into a memory stage, where it is used to define a new memory state.
Thus, the memory stage is simply a mapping :: dim_out_pvs_layer × dim_out_memory -> dim_out_memory. This last output is remembered untill the next evaluation. 
(Thus, simply storing the previous k-outputs is a special case, where dim_out_memory = k* dim_out_pvs_layer
however, the algorithm may also decide to keep track of something else --- such as a moving average of something.)
The next layer receives as input the previous layer as well as the new output of the memory section.
""" 



class Layer():
	def __init__(self, d_in, d_out, init_connections, config, init_params=None):
		"""
		d_in and d_out are input/output dimensions.
		init_connections is supposed to be a list of length d_out containing a lists of input indices for each output.
		"""
		self.d_in = d_in
		self.d_out = d_out
		self.connections = init_connections
		for k, v in config.items():
			setattr(self, k, v)
		self.update()

		# self.act_fn = act_fn

		self.weights = [ np.zeros(d_w) for d_w in self.weight_dims ]
		self.covs_per_output = [ np.random.choice(self.covs_range, size=d_w) for d_w in self.weight_dims ]
		self.act_fns_inds = [ np.random.choice(range(len(self.act_fns)), p=self.act_fns_probs) for _ in range(self.d_out) ]
		self.normal_sampler = u.StandardNormalBatchSampler(d_in)
		self.normal_sampler = u.StandardNormalBatchSampler(10*d_in*d_out)

	def update(self):
		self.not_connected = [ list(set(range(self.d_in)) - set(ix)) for ix in self.connections]
		self.weight_dims = [ len(ix) for ix in self.connections ]

	def get_total_connections(self):
		return np.sum([len(ix) for ix in self.connections])


	def add_inputs(self, n):
		"""
		Don't add any new connections, just change the input-dimension and self.update() so that later calls of "grow" may add the new connections.
		"""
		if self.in_dimension_fixed:
			return
		self.d_in += n
		self.update()

	def add_outputs(self, n):
		if self.out_dimension_fixed:
			return
		self.connections += [[]]*n
		self.weights += [np.zeros(0)]*n
		self.covs_per_output += [np.zeros(0)]*n
		self.act_fns_inds += [ np.random.choice(range(len(self.act_fns)), p=self.act_fns_probs) for _ in range(n) ]
		self.d_out += n
		self.update()


	def remove_inputs(self, n):
		"""
		TODO Keep track of i-th-input-value; x_i * sum_j |w_ij|/y_j to determine the least-useful
		"""
		if self.in_dimension_fixed or n>self.d_in:
			return
		inds = list(np.random.choice(range(self.d_in), size=n, replace=False))
		for j in range(self.d_out):
			# remove the appropriate indices
			locations = [ l for l,c in enumerate(self.connections[j]) if c in inds ]
			self.connections[j] = list(set(self.connections[j]) - set(inds))
			self.weights[j] = np.delete(self.weights[j], locations)
			self.covs_per_output[j] = np.delete(self.covs_per_output[j], locations)

			# Shift back indices appropriately.
			for i in sorted(inds, reverse=True): # start from the highest removed input index
				self.connections[j] = [ x if x < i else x-1 for x in self.connections[j]]
		self.d_in -= n
		self.update()

 
	def remove_outputs(self, n):
		if self.out_dimension_fixed or n>self.d_out:
			return
		inds = np.random.choice(range(self.d_out), size=n, replace=False)
		for i in inds:
			del self.connections[i]
			del self.weights[i]
			del self.covs_per_output[i]
			del self.act_fns_inds[i]
		self.d_out -= n
		self.update()

	def _evaluate(self, x, i):
		"""
		Determine the value of the i-th output dimension, with input x. ;
		at this point, could store the values of |w .* x[i_x]| / np.dot( |w|, |x[ix]| ) 
		or |w .* x[i_x]| /  ||w||^2 * ||x[ix]||^2
		to keep track of the contributions of each connection and inputs
		"""
		ix = self.connections[i]
		w = self.weights[i]
		if not ix: # If list is empty
			return 0
		act_fn = self.act_fns[self.act_fns_inds[i]]
		return act_fn(np.dot(w,x[ix]))


	def evaluate(self, x):
		"""
		todo: See if there is some more efficient method to do this
		"""
		if not self.connections: # if list is empty, means there are no outputs!!
			return np.zeros(0)
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
					self.weights[i] = np.append(self.weights[i], 0) # Set the weight to the new connection to be zero.
					self.covs_per_output[i] = np.append(self.covs_per_output[i], np.random.choice(self.covs_range))	
					self.update()
	def shrink(self):
		"""
		Remove for each output a uniformly random connection with some probability.
		"""
		if self.p_shrink_outer>0 and bernoulli_trial(self.p_shrink_outer):
			for i, ix in enumerate(self.connections):
				while self.p_shrink_inner>0 and ix and bernoulli_trial(self.p_shrink_inner): # try to replace the >1 condition by ".. and ix and ...".
					i_rem = np.random.choice(range(len(ix)))
					self.connections[i].remove(ix[i_rem])
					self.weights[i] = np.delete(self.weights[i], i_rem)
					self.covs_per_output[i] = np.delete(self.covs_per_output[i], i_rem)
					self.update()


	def mutate(self):
		self.grow()
		self.shrink()

	def perturb_params(self):
		"""
		TODO: See if the noise-sampling can be done more efficient, somehow. Would be best to connect it to some real-life noise that acts nearly instantly...
		"""

		for i, covs in enumerate(self.covs_per_output):
			d_w = self.weight_dims[i]
			if d_w==0:
				continue
			self.weights[i] += np.multiply(covs, self.normal_sampler.get(d_w))
			cov_eps = np.random.choice(self.cov_eps, size=d_w)
			self.covs_per_output[i] += np.multiply(cov_eps, self.normal_sampler.get(d_w))



def get_new_layer(d_in, d_out, n_connections, config):
	"""
	Each output is connected to 'n_connections inputs', which are chosen randomly.
	"""
	if n_connections==-1:
		initial_connections = [ list(range(d_in)) for _ in range(d_out) ]
	else:
		initial_connections = [ [np.random.choice(range(d_in),  replace=False)] for _ in range(d_out) ] #TODO: do case -1 separately, much more effective.
	return Layer(d_in, d_out, initial_connections, config)



class NetworkSection():
	def __init__(self, layers, d_in, d_out, config):
		self.layers = layers
		self.d_in = d_in
		self.d_out = d_out
		for k, v in config.items():
			setattr(self, k, v)
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
		raise NotImplementedError

	def _insert_layer_at_position(self, pos):
		d_in = self._n_inputs_of_layer_at_position(pos)
		new_layer = get_new_layer(d_in, self.layers[pos].d_out, 1, self.layer_config)
		self.layers.insert(pos, new_layer)

	def update(self):
		# If a layer has no connections, remove all subsequent layer and make a new full layer with self.d_out outputs.
		i_unconnected = -1
		for i, layer in enumerate(self.layers):
			if not layer.connections:
				i_unconnected = i
		if i_unconnected != -1:
			self._insert_layer_at_position(i_unconnected)
			self.layers = self.layers[:i_unconnected]
			d_in = self._n_inputs_of_layer_at_position(i_unconnected)
			self.layers.append(get_new_layer(d_in, self.d_out, -1, self.layer_config))

		self.n_layers = len(self.layers)
		self.cum_out_dims = np.cumsum([ layer.d_out for layer in self.layers])


	def evaluate(self, inpt):
		output = inpt
		if not self.layers:
			return np.zeros(self.d_out)
		for layer in self.layers:
			output = layer.evaluate(output)
		return output


	def _mutate_nodes(self):
		# Add and remove nodes in between layers
		for i in range(self.n_layers-1):
			first_layer, second_layer = self.layers[i], self.layers[i+1]
			while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				first_layer.add_outputs(1)
				second_layer.add_inputs(1)
			while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
				first_layer.remove_outputs(1)
				second_layer.remove_inputs(1)


	def mutate(self):

		# Mutate layers
		for layer in self.layers:
			layer.mutate()

		self._mutate_nodes()

		# Add new layer at the end
		if self.p_new_layer>0 and bernoulli_trial(self.p_new_layer):
			pos = len(self.layers) - 1 
			d_in = self._n_inputs_of_layer_at_position(pos)
			self.layers.append(get_new_layer(d_in, self.d_out, 1, self.layer_config))

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
		for i in range(self.n_layers-1):
			assert self.layers[i].d_out==self.layers[i+1].d_in, "The dimensions of the input and output of subsequent layers should match"


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

	def _mutate_nodes(self):
		return # for now! Need to be implemented so that all subsequent layers are adapted appropriately.

		# for i in range(self.n_layers-1):
		# 	first_layer, subsequent_layers = self.layers[i], self.layers[i+1:]
		# 	while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
		# 		first_layer.add_outputs(1)
		# 		pos = self._n_inputs_of_layer_at_position(i+1)
		# 		for layer in subsequent_layers:
		# 			layer.add_inputs_at_pos([pos])
		# 	while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
		# 		p = np.random.choice(range(first_layer.d_out))
		# 		first_layer.remove_outputs_at_pos([p])
		# 		pos = self._n_inputs_of_layer_at_position(i)+p
		# 		for layer in subsequent_layers:
		# 			layer.remove_inputs_at_pos([pos])

	def evaluate(self, inpt):
		output = inpt
		if not self.layers:
			return np.zeros(self.d_out)
		for layer in self.layers:
			output = np.concatenate((output, layer.evaluate(output)))
		return output
    

class Evolutionary_NN():
	"""
	TODO:
	Implement different activation functions: step function, logistic, sinc, gaussian, ...? Which make sense to use?

	TODO:
	Make it so that section_1 and section_2 are optional. If not given, it should connect the thing before and after.
	(E.g. if both are not given, directly connect input to final layer.)

	TODO: 
	Test if a normal NN is good at learning logical operations(, perhaps after adding some more activatin functions, in particular the step or logistic functions seem necessary)
	If not implement the logical section, too.

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
	def __init__(self, section_1, section_2, d_out, config):
		self.section_1 = section_1
		self.section_2 = section_2
		self.d_in = section_1.d_in
		self.d_out = d_out
		for k, v in config.items():
			setattr(self, k, v)

		final_layer_config = config['layer_config']
		final_layer_config['out_dimension_fixed'] = True
		final_layer_config['act_fns'] = [ lambda x: x ]
		final_layer_config['act_fns_probs'] = [ 1 ]

		self.final_layer = get_new_layer(self.section_2.d_out, d_out, -1, final_layer_config)
		assert 0<self.p_new_node<1
		assert 0<self.p_remove_node<1
		
		self.avg_rew = RunningAverage(N=self.score_memory)

	def get_total_connections(self):
		return self.section_1.get_total_connections() + self.section_2.get_total_connections() + self.final_layer.get_total_connections()

	def evaluate(self, x):
		"""
		Returns a probability distribution over the output dimensions
		"""
		x = self.section_1.evaluate(x)
		x = self.section_2.evaluate(x)
		x = self.final_layer.evaluate(x)
		return x # temporary
		# return u.softmax(x)

	def sample_output(self, x):
		"""
		Samples from the distribution resulting from self.evaluate. Returns an integer in between (0, self.d_out)
		"""
		return self.evaluate(x)
		# return np.random.choice(range(self.d_out), p=self.evaluate(x))

	def mutate(self):
		self.section_1.mutate()
		self.section_2.mutate()
		self.final_layer.mutate()

		# TODO Make this into a function... 

		# The commented section is broken, because it will not update all subsequent layers in section 2 appropriately.
		# while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
		# 		self.section_1.layers[-1].add_outputs(1)
		# 		self.section_2.layers[0].add_inputs(1)
		# while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
		# 		self.section_1.layers[-1].remove_outputs(1)
		# 		self.section_2.layers[0].remove_inputs(1)

		while self.p_new_node>0 and bernoulli_trial(self.p_new_node):
				self.section_2.layers[-1].add_outputs(1)
				self.final_layer.add_inputs(1)
		while self.p_remove_node>0 and bernoulli_trial(self.p_remove_node):
				self.section_2.layers[-1].remove_outputs(1)
				self.final_layer.remove_inputs(1)


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
	def __init__(self, get_env, population_size, initial_policy):
		self.population_size = population_size
		self._get_env = get_env
		self.envs = [ get_env() for _ in range(population_size)]
		self.states = [ env.reset() for env in self.envs ]
		self.policies = [ deepcopy(initial_policy) for _ in range(population_size) ]


	def train(self, n_steps_total, n_steps_update, mode='average', callbacks=[]):
		"""
		possible modes: 'average' and 'safe', the former uses a moving average and the latter a moving minumum,
		to score various policies. 
		TODO: Implement 'safe' mode.
		"""
		for i in range(n_steps_total):
			for k, (env, pol, state) in enumerate(zip(self.envs, self.policies, self.states)):
				u.timer.start('evaluate')
				action = pol.sample_output(state)
				u.timer.stop('evaluate')
				s_new, r, done, _  = env.step(action)
				r = r if not done else r-1 # This is done for compatibility with tasks of the form "stay alive as long as possible" that give the same award every step.
				pol.avg_rew.update(r)
				self.states[k] = s_new if not done else env.reset()
				
			# Every ... steps, sample a new population from a distribution based on their average scores.
			if (i+1) % n_steps_update==0:
				scores = np.array([ pol.avg_rew.get() for pol in self.policies ])
				scores -= np.min(scores)
				probs = None if np.sum(scores)==0 else scores/np.sum(scores)

				# Cannot use choice(policies, **size=self.population_size**, p=props) because we need to deepcopy the policies.
				self.policies = [ deepcopy(np.random.choice(self.policies, p=probs)) for _ in range(self.population_size) ]

			if (i+1) % n_steps_update==0:
				for pol in self.policies:
					u.timer.start('mutate')
					pol.mutate()
					u.timer.stop('mutate')

					u.timer.start('perturb_params')
					pol.perturb_params()
					u.timer.stop('perturb_params')

			for callback in callbacks:
				callback(self, i)

			if (i+1)%100==0:
				print(i+1)



	@staticmethod
	def _score_single_episode(env, pol, max_steps=-1, render=False):
		s = env.reset()
		done = False
		r_tot = 0
		counter = 0
		while not done and counter!=max_steps:
			if render:
				env.render()
			action = pol.sample_output(s)
			s, r, done, _  = env.step(action)
			r_tot += r
			counter += 1
		env.close()
		return r_tot


	def score(self, n_episodes=1, max_steps=-1, render=False):
		"""
        Returns the total scores over the n episodes.
		"""
		env = self._get_env()
		i = np.argmax([ pol.avg_rew.get() for pol in self.policies ])
		best_policy = self.policies[i]
		return [ EvolutionaryPolicyOptimizer._score_single_episode(env, best_policy, max_steps=max_steps, render=render) for _ in range(n_episodes) ]
