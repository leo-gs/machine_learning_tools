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
	return np.rec.array(best_label, dtype=[('prediction', label_type)])

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
			if data.is_numeric(self.dataset.get_column_datatype(self.attribute)):
				next_node = self.left_node if datapoint[self.attribute] < self.threshold else self.right_node
			else:
				next_node = self.left_node if datapoint[self.attribute] != self.threshold else self.right_node
			return next_node.predict(datapoint)

	def set_leaf_probabilities(self):
		label_counts = objectives.count_labels(self.dataset, self.label)
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

	def __str__(self):
		# for debugging
		string = ''
		remaining = [(0,self.root)]
		while remaining:
			level, node = remaining[0]
			string = string + 'level' + str(level)
			string = string + (', attribute=' + str(node.attribute) if not node.is_leaf() else '')
			string = string + (', threshold=' + str(node.threshold) if not node.is_leaf() else '')
			string = string + ': ' + str(node.dataset.datapoints[self.label]) + '\n'
			if node.left_node:
				remaining.append((level+1, node.left_node))
			if node.right_node:
				remaining.append((level+1, node.right_node))
			del remaining[0]
		return string
			
	def grow(self, depth_remaining, node):
		if depth_remaining == 0:
			node.set_leaf_probabilities()
			return

		datasubset = node.dataset.get_feature_subset(self.label, self.m)
		threshold, attribute, gain = objectives.find_optimal_split(datasubset, self.label)
		if gain == 0:
			node.set_leaf_probabilities()
			return

		node.threshold = threshold
		node.attribute = attribute

		left_data, right_data = objectives.split_data(datasubset, attribute, threshold)

		node.left_node, node.right_node = Node(left_data, self.label), Node(right_data, self.label)

		self.grow(depth_remaining - 1, node.left_node)
		self.grow(depth_remaining - 1, node.right_node)

	def get_probabilities(self, datapoint):
		return self.root.predict(datapoint)

class RandomForest():

	def __init__(self, dataset, label):
		self.dataset = dataset
		self.label = label

	def select_sample_with_replacement(self, dataset=None, sample_size=None):
		if not dataset:
			dataset = self.dataset
		if not sample_size:
			sample_size = dataset.datapoints.shape[0]
		datapoints = np.random.choice(dataset.datapoints, size=sample_size, replace=True)
		return data.DataSet(datapoints=datapoints, datatypes=dataset.datatypes)

	def grow(self, max_depth=1, size=1, m=None):
		if not m:
			m = int(math.sqrt(len(self.dataset.datatypes)))

		self.forest = []
		for i in range(size):
			tree_data = self.select_sample_with_replacement()
			tree = Tree(tree_data, max_depth, self.label, m)
			self.forest.append(tree)

	def predict(self, datapoints):
		# create a new structured array to hold datapoints + predictions
		fields = datapoints.dtype.names
		predictions = np.zeros(datapoints.shape[0], dtype=[(field, datapoints.dtype[field]) for field in fields] + [('prediction', datapoints.dtype[self.label])])

		for index in range(datapoints.shape[0]):
			datapoint = datapoints[index]
			total_probabilities = {}
			for tree in self.forest:
				probabilities = tree.get_probabilities(datapoint)
				for label, probability in probabilities.items():
					total_probabilities[label] = total_probabilities.get(label, 0) + probability
			for label in total_probabilities.keys():
				total_probabilities[label] = total_probabilities[label] / len(self.forest)

			point_prediction = prediction(total_probabilities, datapoints.dtype[self.label])
			
			for field in fields:
				predictions[index][field]

		return predictions
