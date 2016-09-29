import csv
import numpy as np
import random

'''
File format: input is a csv file
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

	def select_columns(self, attributes):
		datapoints = self.datapoints[attributes]
		datatypes = [(name, dtype) for name, dtype in self.datapoints.dtype.fields.items() if name in attributes]
		return DataSet(datapoints=datapoints,)

	def get_feature_subset(self, label, m):
		# add 1 to m to account for label
		m += 1
		attribute_sample = random.sample([dtype for dtype in self.datapoints.dtype.names], m)
		if label not in attribute_sample:
			attribute_sample[-1] = label
		return self.select_columns(attribute_sample)

	# either filename or datapoints and datatypes must be given
	def __init__(self, filename=None, datapoints=None, ratio_validation=None, shuffle=False):
		if not filename and (datapoints is None):
			raise ValueError('Either a csv filename or an array of datapoints and list of datatypes must be given.')

		if datapoints is None:
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

		if shuffle:
			np.random.shuffle(datapoints)

		self.datapoints = datapoints

		if ratio_validation:
			validation_split = int((1-ratio_validation) * datapoints.shape[0]) if datapoints.shape else None
			self.datapoints = datapoints[:validation_split]
			self.validation_datapoints = datapoints[validation_split:]