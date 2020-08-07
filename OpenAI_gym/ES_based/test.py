import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from ev_opt import *



def main():
	shifted_sinc = np.vectorize(lambda x: np.sinc(x-2))
	obj_fun = lambda x: shifted_sinc(x)[:,0]
	population_size=100
	init_params = np.random.uniform(low=-9, high=-8, size=population_size).reshape(population_size, 1)
	covariances = [2**(-x) * np.array([[1]]) for x in range(-4, 4)]
	probs =  [ 1/len(covariances) ] * len(covariances) 
	optimizer = EvolutionaryOptimizer(obj_fun, init_params, covariances, probs)

	fig = plt.figure()
	ax = fig.add_subplot(1,1,1)

	

	def animate(i):
		ax.clear()
		x = np.linspace(-12, 12, 300)
		ax.plot(x,obj_fun(x.reshape(300,1)))
		ax.scatter(optimizer.params[:,0], optimizer.evaluate())
		optimizer.step()


	ani = animation.FuncAnimation(fig, animate)

	plt.show()


if __name__ == '__main__':
	runmin = RunningMinimum()
	for i in reverse(range(10)):
		runmin.update(i)
		print(runmin.current.minimum)
	# main()