import data
import objectives

import math
import numpy as np
from numpy.lib.recfunctions import append_fields
import random

'''
For usage example tests/testcases/test_randomforest.py
'''

def prediction(probabilities, label_type):
	best_probability = -1
	best_label = None
	for label, probability in probabilities.items():
		if probability > best_probability:
			best_probability = probability
			best_label = label

	return np.rec.array((best_label, best_probability), dtype=[('_prediction', label_type), ('_probability', np.dtype(float))])

class Node():

	def __init__(self, dataset, label):
		self.dataset = dataset
		self.label = label
		self.left_node = None
		self.right_node = None

	def is_leaf(self):
		return not self.left_node and not self.right_node

	def predict(self, datapoint):
		if self.is_leaf():
			return self.probabilities
		else:
			next_node = None
			if data.is_numeric(self.dataset.get_attribute_datatype(self.attribute)):
				next_node = self.left_node if datapoint[self.attribute] < self.threshold else self.right_node
			else:
				next_node = self.left_node if datapoint[self.attribute] != self.threshold else self.right_node
			return next_node.predict(datapoint)

	def set_leaf_probabilities(self):
		label_counts = self.dataset.count_labels(self.label)
		total = float(objectives.sum_values(label_counts))
		for label in label_counts.keys():
			label_counts[label] = label_counts[label]/total
		self.probabilities = label_counts

class Tree():

	def __init__(self, root_dataset, max_depth, label, m):
		self.label = label
		self.m = m
		self.root = Node(root_dataset, self.label)
		self.grow(max_depth, self.root)
			
	def grow(self, depth_remaining, node):
		if depth_remaining == 0:
			node.set_leaf_probabilities()
			return

		datasubset = node.dataset.get_feature_projection(self.label, self.m)
		threshold, attribute, gain = objectives.find_optimal_split(datasubset, self.label)
		if gain == 0:
			node.set_leaf_probabilities()
			return

		node.threshold = threshold
		node.attribute = attribute

		left_data, right_data = datasubset.split_data(attribute, threshold)

		node.left_node, node.right_node = Node(left_data, self.label), Node(right_data, self.label)

		self.grow(depth_remaining - 1, node.left_node)
		self.grow(depth_remaining - 1, node.right_node)

	def get_probabilities(self, datapoint):
		return self.root.predict(datapoint)

class RandomForest():

	def __init__(self, dataset, label):
		self.dataset = dataset
		self.label = label

	def grow(self, max_depth=1, size=1, m=None):
		if not m:
			m = int(math.sqrt(len(self.dataset.get_attributes())))

		self.forest = []
		for i in range(size):
			tree_data = self.dataset.get_sample_with_replacement()
			tree = Tree(tree_data, max_depth, self.label, m)
			self.forest.append(tree)

	def predict(self, datapoints):
		# create a new structured array to hold datapoints + predictions
		fields = datapoints.dtype.names
		predictions = np.zeros(datapoints.shape[0], dtype=[(field, datapoints.dtype[field]) for field in fields] + [('_prediction', datapoints.dtype[self.label]), ('_probability', np.dtype(float))])

		for index in range(datapoints.shape[0]):
			datapoint = datapoints[index]

			# calculate label probabilites and prediction
			total_probabilities = {}
			for tree in self.forest:
				probabilities = tree.get_probabilities(datapoint)
				for label, probability in probabilities.items():
					total_probabilities[label] = total_probabilities.get(label, 0) + probability
			for label in total_probabilities.keys():
				total_probabilities[label] = total_probabilities[label] / len(self.forest)

			point_prediction = prediction(total_probabilities, datapoints.dtype[self.label])
			
			for field in fields:
				predictions[index][field] = datapoint[field]
			predictions[index]['_prediction'] = point_prediction['_prediction']
			predictions[index]['_probability'] = point_prediction['_probability']

		return predictions
