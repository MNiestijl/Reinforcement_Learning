import matplotlib.pyplot as plt
import numpy as np

def plot_improvement(agent, n_episodes, sample_size, train_size=1,ax=None, figure=-1):

	def func(sample):
		print("Training episode {}".format(sample))
		for i in range(train_size):
			agent.train()
		return agent.scores(sample_size)

	all_scores = list(map(func, range(n_episodes)))
	averages = list(map(np.mean, all_scores))

	if ax is None:
		fig = plt.figure(figure) if figure>-1 else plt.figure()
		ax = fig.add_subplot(1,1,1)
	ax.plot(list(map( lambda i: int(train_size*i), range(1,n_episodes+1))), averages)
	ax.set_title('Improvement of various agents')
	ax.set_xlabel('Episode')
	ax.set_ylabel('Average total reward over {} trails'.format(sample_size))
	ax.xaxis.label.set_fontsize(16)
	ax.yaxis.label.set_fontsize(16)

def plot_improvements(agents, n_episodes, sample_size, figure=0, train_size=1):
	fig = plt.figure(figure)
	ax = fig.add_subplot(1,1,1)
	for _, agent in agents:
		plot_improvement(agent, n_episodes, sample_size, train_size, ax=ax)
	ax.legend(list(map( lambda agent: agent[0], agents)))

def plot_histogram(agent, n_episodes, figure=0, ax=None):
	scores = agent.scores(n_episodes)
	if ax is None:
		fig = plt.figure(figure)
		ax = fig.add_subplot(1,1,1)
	ax.hist(scores, alpha=0.5)

def plot_histograms(agents, n_episodes, figure=-1):
	fig = plt.figure(figure) if figure>-1 else plt.figure()
	ax = fig.add_subplot(1,1,1)
	for _, agent in agents:
		plot_histogram(agent, n_episodes, figure=figure, ax=ax)
	ax.legend(list(map( lambda agent: agent[0], agents)))

