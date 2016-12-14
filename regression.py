import data

import numpy as np

class LinearRegression():

	def __init__(self, dataset):
		self.dataset = dataset

	def least_squares(self, predictor, label):
		if not data.is_numeric(self.dataset.get_attribute_datatype(predictor)) or not data.is_numeric(self.dataset.get_attribute_datatype(label)):
			raise ValueError('Both the predictor and label must be numeric')

		n = len(self.dataset)
		xs = self.dataset.get_column_vector(predictor)
		ys = self.dataset.get_column_vector(label)

		mean_x = xs.mean()
		mean_y = ys.mean()

		b1 = ((xs - mean_x) * (ys - mean_y)).sum() / ((xs - mean_x) ** 2).sum()
		b0 = (float(ys.sum()) / n) - b1 * (float(xs.sum()) / n)

		return (b1, b0)