import data

import math
import numpy as np

def sum_values(counts):
	total = 0
	for count in counts.values():
		total += count
	return float(total)

def entropy(counts, size=None):
	if not size:
		size = sum_values(counts)
	total_entropy = 0
	for count in counts.values():
		p = float(count)/size
		entropy = -1 * p * math.log(p, 2)
		total_entropy += entropy
	return total_entropy

def split_data(dataset, attribute, threshold):
	# if attribute represents a numeric datatype, split into < threshold and >= threshold
	# otherwise split into != threshold and == threshold
	# not efficient for multiple splits
	attribute_datatype = dataset.datapoints.dtype[attribute]
	split_datapoints = None
	if data.is_numeric(attribute_datatype):
		split_datapoints = (dataset.datapoints[dataset.datapoints[attribute] < threshold], dataset.datapoints[dataset.datapoints[attribute] >= threshold])
	else:
		split_datapoints = (dataset.datapoints[dataset.datapoints[attribute] != threshold], dataset.datapoints[dataset.datapoints[attribute] == threshold])
	return (data.DataSet(datapoints=split_datapoints[0]), data.DataSet(datapoints=split_datapoints[1]))

def information_gain(counts, left_counts, right_counts):
	total_size = sum_values(counts)
	left_size = sum_values(left_counts)
	right_size = sum_values(right_counts)
	if total_size == 0 or left_size == 0 or right_size == 0:
		return 0
	return entropy(counts, size=total_size) - (left_size/total_size) * entropy(left_counts, size=left_size) - (right_size/total_size) * entropy(right_counts, size=right_size)

def find_optimal_split(dataset, label, attribute=None):
	# if attribute is not given, all attributes will be searched
	# copy datapoints so original data is not modified
	datapoints = np.copy(dataset.datapoints)
	counts = dataset.count_labels(label)
	attributes = [attribute]
	if not attribute:
		attributes = dataset.datapoints.dtype.names

	optimal_threshold = None
	optimal_attribute = None
	optimal_gain = 0

	for attribute in attributes:
		# don't use the label as a splitting attribute
		if attribute == label:
			continue

		datapoints = np.sort(datapoints, order=attribute)
		# "left" and "right" representing data on either side of a given split
		# keeping a running tally is cheaper than calling split_data multiple times
		left_counts = dataset.count_labels(label)
		right_counts = {}

		index = datapoints.shape[0]-1
		while index >= 0:
			threshold = datapoints[index][attribute]
			point_label = datapoints[index][label]
			right_counts[point_label] = right_counts.get(point_label, 0) + 1
			left_counts[point_label] = left_counts[point_label] - 1
			if left_counts[point_label] == 0:
				del left_counts[point_label]
			index -= 1

			# handle duplicate attribute values
			while index >= 0 and datapoints[index][attribute] == threshold:
				point_label = datapoints[index][label]
				right_counts[point_label] = right_counts.get(point_label, 0) + 1
				left_counts[point_label] = left_counts[point_label] - 1
				if left_counts[point_label] == 0:
					del left_counts[point_label]
				index -= 1

			split_gain = information_gain(counts, left_counts, right_counts)
			# print split_gain
			# print left_counts
			# print right_counts
			# print '\n'
			if split_gain > optimal_gain:
				optimal_threshold = threshold
				optimal_attribute = attribute
				optimal_gain = split_gain

	return (optimal_threshold, optimal_attribute, optimal_gain)
		