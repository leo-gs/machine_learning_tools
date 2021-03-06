import csv
import numpy as np
import random

'''
File format: input is a csv file or sqlite3 file
First row: data attribute titles (should be unique)
Second row: data types
Each following row represents a data point (each column contains data for a attribute/label)

Data Types: int, float, boolean, text
Booleans can be represented as true/false or X/[empty cell]
'''

def convert_str_to_np_type(type_str):
	types_dict = {
		'int': np.dtype(int),
		'float': np.dtype(float),
		'boolean': np.dtype(bool),
		'text': np.dtype(object)
	}
	return types_dict[type_str]

def is_numeric(attribute_datatype):
	return attribute_datatype == str(np.dtype(int)) or attribute_datatype == str(np.dtype(float))

class DataSet():

	# either filename or datapoints and datatypes must be given
	def __init__(self, filename=None, datapoints=None, ratio_validation=None, shuffle=False):
		if not filename and (datapoints is None):
			raise ValueError('Either a filename or an array of datapoints and list of datatypes must be given.')

		if datapoints is None:
			if filename.split('.')[-1] == 'csv':
				datapoints = self.init_data_from_csv(filename)

		if shuffle:
			np.random.shuffle(datapoints)

		self.datapoints = datapoints

		if ratio_validation:
			validation_split = int((1-ratio_validation) * datapoints.shape[0]) if datapoints.shape else None
			self.datapoints = datapoints[:validation_split]
			self.validation_datapoints = datapoints[validation_split:]

	def select_columns(self, attributes):
		datapoints = self.datapoints[attributes]
		datatypes = [(name, dtype) for name, dtype in self.datapoints.dtype.fields.items() if name in attributes]
		return DataSet(datapoints=datapoints,)

	def get_random_feature_projection(self, label, m):
		# add 1 to m to account for label
		m += 1
		attribute_sample = random.sample([dtype for dtype in self.datapoints.dtype.names], m)
		if label not in attribute_sample:
			attribute_sample[-1] = label
		return self.select_columns(attribute_sample)

	def count_labels(self, label):
		# label is a the name of a column in the spreadsheet
		# this counts the number of occurences of each label value
		counts = {}
		for point in self.datapoints:
			counts[point[label]] = counts.get(point[label], 0) + 1
		return counts

	def split_data(self, attribute, threshold):
		# if attribute represents a numeric datatype, split into < threshold and >= threshold
		# otherwise split into != threshold and == threshold
		# not efficient for multiple splits
		attribute_datatype = self.datapoints.dtype[attribute]
		split_datapoints = None
		if is_numeric(attribute_datatype):
			split_datapoints = (self.datapoints[self.datapoints[attribute] < threshold], self.datapoints[self.datapoints[attribute] >= threshold])
		else:
			split_datapoints = (self.datapoints[self.datapoints[attribute] != threshold], self.datapoints[self.datapoints[attribute] == threshold])
		return (DataSet(datapoints=split_datapoints[0]), DataSet(datapoints=split_datapoints[1]))

	def init_data_from_csv(self, filename):
		# Parses data from a csv spreadsheet
		reader = csv.reader(open(filename, 'rU'))
		# The first row should be the column titles/attributes
		# The second row should be the types of the column's data ('int', 'float', 'boolean', 'text')
		attributes = reader.next()
		dtypes = [str(convert_str_to_np_type(elt)) for elt in reader.next()]
		datatypes = [(attr, dtype) for attr, dtype in zip(attributes, dtypes)]
		bool_indices = [i for i in range(len(dtypes)) if dtypes[i] == np.dtype(bool)]

		def handle_booleans(elt):
			for index in bool_indices:
				elt[index] = elt[index].lower() == 'true' or elt[index].lower() == 'x' or elt[index].lower() == 'yes'
			return elt

		# Reads each row as a datapoint, with each column representing an attribute/label
		data = [tuple(handle_booleans(elt)) for elt in list(reader)]
		datapoints = np.array(data, dtype=datatypes)
		return datapoints

	def get_attribute_datatype(self, attribute):
		return self.datapoints.dtype[attribute]

	def get_attributes(self):
		return self.datapoints.dtype.names

	def get_column_vector(self, attribute):
		return np.array(self.datapoints[attribute])

	# return number of points in dataset.datapoints
	def __len__(self):
		return self.datapoints.shape[0]

	# selects a sample with replacement from the dataset
	# if no sample size is given, the sample size will be the size of the entire dataset
	def get_sample_with_replacement(self, dataset=None, sample_size=None):
		if not dataset:
			dataset = self
		if not sample_size:
			sample_size = self.datapoints.shape[0]
		datapoints = np.random.choice(self.datapoints, size=sample_size, replace=True)
		return DataSet(datapoints=datapoints)

	def get_sorted_iterator(self, attribute=None):
		# iterates through backwards
		return self.SortedIterator(self.datapoints, attribute)

	class SortedIterator():
		# if attribute == None, data will not be sorted.
		def __init__(self, datapoints, attribute):
			self.datapoints = np.sort(np.copy(datapoints), order=attribute) if attribute else datapoints # copy datapoints so original data is not modified
			self.index = datapoints.shape[0]-1

		def __iter__(self):
			return self

		def next(self):
			point = self.datapoints[self.index]
			self.index -= 1
			return point

		def peek(self):
			return self.datapoints[self.index]

		def has_more(self):
			return self.index >= 0

