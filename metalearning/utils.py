import time
import traceback
import numpy as np
from load_data import Load_Data


def load_file(file_name):
	load = Load_Data(file_name)
	X, y, categorical = load.split_x_y()
	print X.shape, y.shape
	return X, y, categorical


def load_file_landmarking(file_name):
	load = Load_Data(file_name)
	X, y, categorical = load.split_x_y_landmarking()
	print X.shape, y.shape
	return X, y, categorical


def data_types(X, categorical):
	cols = X.columns.values

	cat_var = []
	num_var = []

	for i in range(len(categorical)):
		if categorical[i] is True:
			cat_var.append(cols[i])
		else:
			num_var.append(cols[i])

	return num_var, cat_var


def input_check(X, y):
	assert X.shape[0] == y.shape[0], "X and y do not have the same shape"
	return


class stopwatch:
	def __enter__(self):
		self.start = time.clock()
		return self

	def __exit__(self, type, value, tb):
		self.end = time.clock()
		self.duration = self.end - self.start

	def elapsed_time(self):
		if self.end is None:
			return time.clock() - self.start
		else:
			return self.duration

