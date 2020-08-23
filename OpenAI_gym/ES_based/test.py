import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..')) # to acces parent folder for imports.

import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ES_based.ev_opt import *
from collections import deque
import utils as u


class EvPolOptTrackerCallback():

	def __init__(self):
		self.max_rewards = deque()
		self.avg_rewards = deque()
		self.avg_connections = deque()

	def __call__(self, ev_pol_opt, i):
		avg_rews = [ pol.avg_rew.get() for pol in ev_pol_opt.policies ]
		avg_connections = np.mean([ pol.get_total_connections() for pol in ev_pol_opt.policies ])
		self.max_rewards.append(np.max(avg_rews))
		self.avg_rewards.append(np.mean(avg_rews))
		self.avg_connections.append(avg_connections)

		if i%50==0:
			print('\nMaximal avg reward: {}'.format(np.max(avg_rews)))
			print('Average reward: {}'.format(np.mean(avg_rews)))
			print('Average connections: {}'.format(avg_connections))


class Test():
	def __init__(self):
		self.xs = [ np.arange(4).reshape(2,2), np.arange(9).reshape(3,3) ]

	def perform(self):
		for i in range(len(self.xs)):
			self.xs[i] = np.arange(16).reshape(4,4)


def test():
	dim_in = int(1e3)
	dw = int(1e3)
	dim_out = int(1e3)
	repeat = 100

	weight_dims = [ dw for _ in range(dim_out) ]
	cells_start = [0] + list(np.cumsum(weight_dims))[:-1]
	
	xs = np.arange(dim_in)
	weights = [ np.arange(weight_dims[i]) for i in range(dim_out) ]
	connections = [ np.arange(weight_dims[i]) for i in range(dim_out) ]

	xs_flat = np.concatenate([ xs[ix] for ix in connections ])
	weights_flat = np.concatenate(weights)

	f_1 = lambda x: max(0,x)
	f_2 = np.vectorize(lambda x: max(0,x))
	
	for _  in range(repeat):
		u.timer.start('method 1')
		result_1 = np.array([ f_1(np.dot(w,xs[ix])) for w, ix in zip(weights, connections) ])
		u.timer.stop('method 1')

		u.timer.start('method 2')
		mult = np.multiply(xs_flat, weights_flat)
		result_2 = f_2(np.add.reduceat(mult, cells_start))
		u.timer.stop('method 2')


	print("Total value of timers:\n")
	for name, times in u.timer.times.items():
		print(name + ": {}".format(np.sum(times)))



def main():

	# Environment specific
	get_env = lambda : gym.make('CartPole-v0')
	d_in = 4
	d_intermediate_1 = 4
	d_intermediate_2 = 4
	d_out = 2

	#Settings
	
	population_size = 100
	train_n_steps = 10000
	n_steps_update = 20


	layer_config = {
		'act_fns' : [ u.relu, u.heaviside ],
		'act_fns_probs' : [ 0.8, 0.2 ],
		'p_grow_outer' : 5e-1,
		'p_grow_inner' : 5e-1,
		'p_shrink_outer' : 5e-1,
		'p_shrink_inner' : 1.01e-1,
		'out_dimension_fixed' : False,
		'in_dimension_fixed' : False,
		'covs_range' : [ 2**x for x in range(-15,6) ],
		'cov_eps' : [ 2**x for x in range(-13,4) ],
	}

	section_config = {
		'p_new_node' : 0.01,
		'p_remove_node' : 0.01,
		'p_new_layer' : 1e-3,
		'layer_config' : layer_config,
	}

	config = {
		'p_new_node' : section_config['p_new_node'],
		'p_remove_node' : section_config['p_remove_node'],
		# 'section_config' : section_config,
		'layer_config' : layer_config,
		'score_memory' : 400

	}


	# p_new_node = 0.1
	# p_remove_node=0.1
	# p_new_layer = 2e-3
	# p_grow_outer = 5e-1
	# p_grow_inner = 5e-1
	# p_shrink_outer = 5e-1
	# p_shrink_inner = 1.01e-1

	# Define optimizer

	layer_sec_1 = get_new_layer(d_in, d_intermediate_1, -1, layer_config)
	layer_sec_2 = get_new_layer(d_intermediate_1, d_intermediate_2, -1, layer_config)

	section_1 = NetworkSection1([layer_sec_1], d_in, d_intermediate_1, section_config)
	section_2 = NetworkSection2([layer_sec_2], d_intermediate_1, d_intermediate_2, section_config)
	init_network = Evolutionary_NN(section_1, section_2, d_out, config)
	# init_network = Evolutionary_NN(section_1, section_2, d_out, p_new_layer=p_new_layer, p_new_node=p_new_node, p_remove_node=p_remove_node,
	# 	p_grow_outer=p_grow_outer, p_shrink_outer=p_shrink_outer, p_grow_inner=p_grow_inner, p_shrink_inner=p_shrink_inner, score_memory=score_memory)
	optimizer = EvolutionaryPolicyOptimizer(get_env, population_size, init_network)

	callback = EvPolOptTrackerCallback()

	# Train
	try:
		optimizer.train(train_n_steps, n_steps_update=n_steps_update, callbacks= [ callback ])
	except KeyboardInterrupt:
		pass
	
	while input("Score? y/n\n")=='y':
		optimizer.score(render=True)


	print("Average value of timers:\n")
	for name, times in u.timer.times.items():
		print(name + ": {}".format(np.mean(times)))

	print("Total value of timers:\n")
	for name, times in u.timer.times.items():
		print(name + ": {}".format(np.sum(times)))


	# fig = plt.figure(2)
	# ax_1 = fig.add_subplot(1,2,1)
	# ax_2 = fig.add_subplot(1,2,2)

	# def animate(i):
	# 	ax_1.clear()
	# 	x = np.linspace(-12, 12, 300)
	# 	ax_1.plot(x,obj_fun(x.reshape(300,1)))
	# 	ax_1 .scatter(optimizer.params[:,0], optimizer.evaluate())


	# ani = animation.FuncAnimation(fig, animate)

	# plt.show()


if __name__ == '__main__':
	test()
	# main()