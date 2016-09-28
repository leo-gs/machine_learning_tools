import sys
sys.path.append('../../')

import data

import numpy as np
import unittest

class TestData(unittest.TestCase):

	def setUp(self):
		self.intdata_fname = '../testdata/test_integerdata.csv'
		self.intdata_dtypes = [('attr1', 'int64'), ('attr2', 'int64'), ('attr3', 'int64')]
		self.intdata_shape = (3,)
		self.mixeddata_fname = '../testdata/test_mixeddata.csv'
		self.mixeddata_dtypes = [('int', 'int64'), ('float', 'float64'), ('text','object'), ('boolean','bool')]
		self.mixeddata_shape = (4,)
		self.performance_fname = '../testdata/test_performancedata.csv'

	def testIntegerData(self):
		testdataset = data.DataSet(self.intdata_fname)
		expected = np.array([(1,0,2),(5,3,3),(2,1,1)], dtype=self.intdata_dtypes)

		self.assertEqual(testdataset.datapoints.dtype, self.intdata_dtypes)
		self.assertEqual(testdataset.datapoints.shape, self.intdata_shape)
		self.assertTrue(np.array_equal(testdataset.datapoints, expected))

	def testMixedData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		expected = np.array([(1,0.5,'hi',True),(1,0.5,'hello',False),(7,0.000,'hello',False),(-4,99.1,'hi',True)], dtype=self.mixeddata_dtypes)

		self.assertEqual(testdataset.datapoints.dtype, self.mixeddata_dtypes)
		self.assertEqual(testdataset.datapoints.shape, self.mixeddata_shape)
		self.assertTrue(np.array_equal(testdataset.datapoints, expected))

	def testInit_givenData(self):
		datapoints = np.array([(1,0,2),(5,3,3),(2,1,1)], dtype=self.intdata_dtypes)
		datatypes = self.intdata_dtypes
		testdataset = data.DataSet(datapoints=datapoints)
		expected = data.DataSet(self.intdata_fname)

		self.assertTrue(np.array_equal(testdataset.datapoints, expected.datapoints))
		self.assertEqual(testdataset.datapoints.dtype, expected.datapoints.dtype)

	def testInit_wrongParameters(self):
		self.assertRaises(ValueError, lambda: data.DataSet(filename=None, datapoints=None))

	def testValidationMixedData(self):
		testdataset = data.DataSet(self.mixeddata_fname, ratio_validation=0.25)

		self.assertEqual(testdataset.datapoints.shape, (3,))
		self.assertEqual(testdataset.validation_datapoints.shape, (1,))

	def testShuffleMixedData(self):
		testdataset = data.DataSet(self.mixeddata_fname, ratio_validation=0.25, shuffle=True)

	def testGetFeatureSubset(self):
		testdataset = data.DataSet(self.mixeddata_fname).get_feature_subset('boolean', 2)

		self.assertEqual(len(testdataset.datapoints[0]), 3)
		self.assertEqual(len(testdataset.datapoints.dtype), 3)
		self.assertTrue('boolean' in testdataset.datapoints.dtype.fields.keys())

	def testPerformance(self):
		testdataset = data.DataSet(self.performance_fname)

if __name__ == '__main__':
	unittest.main()