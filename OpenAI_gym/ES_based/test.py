import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ev_opt import *
from collections import deque


class EvPolOptTrackerCallback():

	def __init__(self):
		self.avg_rewards = deque()
		self.avg_connections = deque()

	def __call__(self, ev_pol_opt):
		total_avg = np.mean([ avg.get() for avg in ev_pol_opt.rewards_avgs ])
		avg_connections = np.mean([ pol.get_total_connections() for pol in ev_pol_opt.policies ])
		self.avg_rewards.append(total_avg)
		self.avg_connections.append(avg_connections)


def main():

	# Environment specific
	get_env = lambda : gym.make('CartPole-v0')
	d_in = 4
	d_intermediate_1 = 4
	d_intermediate_2 = 4
	d_out = 2

	#Settings
	population_size = 100
	train_n_steps = 100
	n_steps_update = 10

	p_new_node = 0.01
	p_remove_node=0.012
	p_new_layer = 0.001
	p_grow_outer = 0.1
	p_grow_inner = 0.1
	p_shrink_outer = 0.12
	p_shrink_inner = 0.12

	# Define optimizer

	layer_sec_1 = get_new_layer(d_in, d_intermediate_1, p_grow_outer=p_grow_outer, p_shrink_outer=p_shrink_outer, p_grow_inner=p_grow_inner, p_shrink_inner=p_shrink_inner)
	layer_sec_2 = get_new_layer(d_intermediate_1, d_intermediate_2, p_grow_outer=p_grow_outer, p_shrink_outer=p_shrink_outer, p_grow_inner=p_grow_inner, p_shrink_inner=p_shrink_inner)

	section_1 = NetworkSection1([layer_sec_1], d_in, d_intermediate_1, p_new_layer=p_new_layer, p_new_node=p_new_node, p_remove_node=p_remove_node)
	section_2 = NetworkSection2([layer_sec_2], d_intermediate_1, d_intermediate_2, p_new_layer=p_new_layer, p_new_node=p_new_node, p_remove_node=p_remove_node)
	init_network = Evolutionary_NN(section_1, section_2, d_out, p_new_node=p_new_node, p_remove_node=p_remove_node)
	optimizer = EvolutionaryPolicyOptimizer(get_env, population_size, init_network, score_memory=100)

	# Train

	optimizer.train(train_n_steps, n_steps_update=n_steps_update)


	fig = plt.figure()
	ax_1 = fig.add_subplot(1,2,1)
	ax_2 = fig.add_subplot(1,2,2)

	def animate(i):
		ax_1.clear()
		x = np.linspace(-12, 12, 300)
		ax_1.plot(x,obj_fun(x.reshape(300,1)))
		ax_1 .scatter(optimizer.params[:,0], optimizer.evaluate())


	# ani = animation.FuncAnimation(fig, animate)

	plt.show()


if __name__ == '__main__':
	main()