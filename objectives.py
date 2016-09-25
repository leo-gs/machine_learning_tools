import data

import math
import numpy as np

def count_labels(dataset, label):
	# label is a the name of a column in the spreadsheet
	counts = {}
	for point in dataset.datapoints:
		counts[point[label]] = counts.get(point[label], 0) + 1
	return counts

def entropy(counts):
	total = 0
	for count in counts.values():
		total += count

	total_entropy = 0
	for count in counts.values():
		p = float(count)/total
		entropy = -1 * p * math.log(p, 2)
		total_entropy += entropy

	return total_entropy

def split_data(dataset, attribute, threshold):
	# if attribute represents a numeric datatype, split into <= threshold and > threshold
	# otherwise split into != threshold and == threshold
	# not efficient for multiple splits
	attribute_datatype = dataset.get_column_datatype(attribute)
	split_datapoints = None
	if attribute_datatype == str(np.dtype(int)) or attribute_datatype == str(np.dtype(float)):
		split_datapoints = (dataset.datapoints[dataset.datapoints[attribute] <= threshold], dataset.datapoints[dataset.datapoints[attribute] > threshold])
	else:
		split_datapoints = (dataset.datapoints[dataset.datapoints[attribute] != threshold], dataset.datapoints[dataset.datapoints[attribute] == threshold])
	return (data.DataSet(datapoints=split_datapoints[0], datatypes=dataset.datatypes), data.DataSet(datapoints=split_datapoints[1], datatypes=dataset.datatypes))

def information_gain(counts, left_counts, right_counts):
	return entropy(counts) - entropy(left_counts) - entropy(right_counts)

def find_optimal_split(dataset, label, attribute=None):
	# if attribute is not given, all attributes will be searched
	# copy datapoints so original data is not modified
	datapoints = np.copy(dataset.datapoints)
	counts = count_labels(dataset, label)
	attributes = [attribute]
	if not attribute:
		attributes = [elt[0] for elt in dataset.datatypes]

	optimal_threshold = None
	optimal_attribute = None
	optimal_gain = 0

	for attribute in attributes:
		if attributes == label:
			continue
		datapoints = np.sort(datapoints, order=attribute)
		# "left" and "right" representing data on either side of a given split
		# keeping a running tally is cheaper than calling split_data multiple times
		left_counts = {}
		right_counts = count_labels(dataset, label)

		index = 0
		while index < datapoints.shape[0]:
			threshold = datapoints[index][attribute]
			point_label = datapoints[index][label]
			left_counts[point_label] = left_counts.get(point_label, 0) + 1
			right_counts[point_label] = right_counts[point_label] - 1
			if right_counts[point_label] == 0:
				del right_counts[point_label]
			index += 1

			# handle duplicate attribute values
			while index < datapoints.shape[0] and datapoints[index][attribute] == threshold:
				point_label = datapoints[index][label]
				left_counts[point_label] = left_counts.get(point_label, 0) + 1
				right_counts[point_label] = right_counts[point_label] - 1
				if  right_counts[point_label] == 0:
					del right_counts[point_label]
				index += 1

			split_gain = information_gain(counts, left_counts, right_counts)
			if split_gain > optimal_gain:
				optimal_threshold = threshold
				optimal_attribute = attribute
				optimal_gain = split_gain

	return (optimal_threshold, optimal_attribute, optimal_gain)
		