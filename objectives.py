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

def information_gain(counts, left_counts, right_counts):
	total_size = sum_values(counts)
	left_size = sum_values(left_counts)
	right_size = sum_values(right_counts)
	if total_size == 0 or left_size == 0 or right_size == 0:
		return 0
	return entropy(counts, size=total_size) - (left_size/total_size) * entropy(left_counts, size=left_size) - (right_size/total_size) * entropy(right_counts, size=right_size)

def find_optimal_split(dataset, label, attribute=None):
	# if attribute is not given, all attributes will be searched
	counts = dataset.count_labels(label)
	attributes = [attribute] # if we already know what attribute to split on
	if not attribute:
		attributes = dataset.get_attributes()

	optimal_threshold = None
	optimal_attribute = None
	optimal_gain = 0

	for attribute in attributes:
		# don't use the label as a splitting attribute
		if attribute == label:
			continue

		# "left" and "right" representing data on either side of a given split
		# keeping a running tally is cheaper than calling split_data multiple times
		left_counts, right_counts = dataset.count_labels(label), {}

		iterator = dataset.get_sorted_iterator(attribute)

		while iterator.has_more():
			point = iterator.next()
			threshold = point[attribute]
			point_label = point[label]
			right_counts[point_label] = right_counts.get(point_label, 0) + 1
			left_counts[point_label] = left_counts[point_label] - 1
			if left_counts[point_label] == 0:
				del left_counts[point_label]

			# handle duplicate attribute values
			while iterator.has_more() and iterator.peek()[attribute] == threshold:
				point_label = iterator.next()[label]
				right_counts[point_label] = right_counts.get(point_label, 0) + 1
				left_counts[point_label] = left_counts[point_label] - 1
				if left_counts[point_label] == 0:
					del left_counts[point_label]

			split_gain = information_gain(counts, left_counts, right_counts)

			if split_gain > optimal_gain:
				optimal_threshold = threshold
				optimal_attribute = attribute
				optimal_gain = split_gain

	return (optimal_threshold, optimal_attribute, optimal_gain)
		