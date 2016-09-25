import csv
import numpy as np

'''
File format: input is a csv file
First row: data attribute titles (should be unique)
Second row: data types
Each following row represents a data point (each column contains data for a attribute/label)

Data Types: int, float, boolean, text
Booleans: a True value is represented by any non-empty string; a False value is represented by an empty string
'''

class DataSet():

	def convert_str_to_np_type(self, type_str):
		types_dict = {
			'int': np.dtype(int),
			'float': np.dtype(float),
			'boolean': np.dtype(bool),
			'text': np.dtype(object)
		}
		return types_dict[type_str]

	def get_column_datatype(self, label):
		return next(elt for elt in self.datatypes if elt[0] == label)[1]

	# either filename or datapoints and datatypes must be given
	def __init__(self, filename=None, datapoints=None, datatypes=None, ratio_validation=0, shuffle=False):
		if not filename and (datapoints is None or not datatypes):
			raise ValueError('Either a csv filename or an array of datapoints and list of datatypes must be given.')

		if datapoints is None:
			# Parses data from a csv spreadsheet
			reader = csv.reader(open(filename, 'rU'))
			# The first row should be the column titles/attributes
			# The second row should be the types of the column's data ('int', 'float', 'boolean', 'text')
			attributes = reader.next()
			dtypes = [str(self.convert_str_to_np_type(elt)) for elt in reader.next()]
			datatypes = [(attr, dtype) for attr, dtype in zip(attributes, dtypes)]
			# Reads each row as a datapoint, with each column representing an attribute/label
			data = [tuple(elt) for elt in list(reader)]
			datapoints = np.array(data, dtype=datatypes)

		if shuffle:
			np.random.shuffle(datapoints)

		validation_split = (1-ratio_validation) * datapoints.shape[0]

		self.datatypes = datatypes
		self.datapoints = datapoints[:validation_split]
		self.validation_datapoints = datapoints[validation_split:]