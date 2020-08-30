import numpy as np
import math as m
import time
from collections import deque
import pdb

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


