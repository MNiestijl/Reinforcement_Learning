import numpy as np
import math as m


def softmax(xs):
	xs = list(map(lambda x: m.exp(x-max(xs)), xs))
	total = sum(xs)
	return np.array(list(map(lambda x: x/total, xs)))
