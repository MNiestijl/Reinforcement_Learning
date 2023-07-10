import numpy as np
import math as m
import time
from collections import deque
import pdb




def sample_indices_from_2D_array(p, axis=1):
	"""
	sample indices from the probability distributions of a 2D np array consisting of probability distributions
	along a specified axis.
	"""
	c = p.cumsum(axis=axis) # Cummulative distributions
	u = np.random.rand(len(c), 1) # Uniformly distributed samples
	return (u < c).argmax(axis=axis) # select the index of the first value of True in (u < c) along the specified axis.

def sample_from_2D_array(p, axis=1):
	"""
	sample from the probability distributions of a 2D np.array consisting of probability distributions
	along a specified axis.
	"""
	n = p.shape[1-axis] # dimension of p along the other axis.
	choices = sample_indices_from_2D_array(p, axis=axis)
	return np.array([ p[i,choices[i]] for i in range(n) ])





class RunningAverage():
	def __init__(self, width):
		self.width = width
		self.avg = 0
		self.numbers = deque(maxlen=width)

	def add(self,x):
		n = len(self.numbers)
		if n == self.width:
			self.avg += (1/self.width) * (x - self.numbers[0])
			self.numbers.append(x)
		else:
			self.avg = (x + n * self.avg)/(n+1)
			self.numbers.append(x)

	def __call__(self):
		return self.avg




def softmax(xs):
	xs = list(map(lambda x: m.exp(x-max(xs)), xs))
	total = sum(xs)
	return np.array(list(map(lambda x: x/total, xs)))

def heaviside(x):
	return 0 if x<= 1 else 1

def relu(x):
	return max(0,x)


class StandardNormalBatchSampler():
	def __init__(self, batch_size):
		self.batch_size = batch_size
		self.refresh()

	def _n_new_max(self):
		return self.batch_size - self.new_ix

	def refresh(self):
		self.batch = np.random.normal(size=self.batch_size)
		self.new_ix = 0

	def get(self, size=1):
		if size>self.batch_size:
			return np.random.normal(size=size)
		result = np.zeros(size)
		i = 0
		size=int(size)
		current_size = 0
		while current_size != size:
			n_new = min(size-current_size, self._n_new_max())
			result[i:i+n_new] = self.batch[self.new_ix:self.new_ix + n_new]
			current_size += n_new
			i += n_new
			self.new_ix += n_new
			if self.new_ix==self.batch_size:
				self.refresh()
		return result



class Timer():

	def __init__(self):
		self.times = {}

	def start(self, name):
		self.start_time = { name: time.time() }

	def stop(self, name):
		self.times.setdefault(name, deque())
		self.times[name].append(time.time() - self.start_time[name])
		del self.start_time[name]

	def mean(self, name):
		return np.mean(self.times[name])



timer = Timer()


