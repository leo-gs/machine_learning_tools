import sys
sys.path.append('../../')

import data
import regression

import numpy as np
import unittest

class TestObjectives(unittest.TestCase):

	def setUp(self):
		self.easy_fname = '../testdata/test_easyleastsquareslinear.csv'
		self.nonnumeric_fname = '../testdata/test_nonnumeric.csv'
		self.standard_fname = '../testdata/test_leastsquares.csv'

	def testLeastSquares_nonNumeric(self):
		testdataset = data.DataSet(self.nonnumeric_fname)
		testregression = regression.LinearRegression(testdataset)

		self.assertRaises(ValueError, testregression.least_squares, 'text1', 'text2')

	def testLeastSquares_easy(self):
		testdataset = data.DataSet(self.easy_fname)
		testregression = regression.LinearRegression(testdataset)

		b1,b0 = testregression.least_squares('predictor', 'label')
		self.assertEqual(int(b1), 1)
		self.assertEqual(int(b0), 0)

	def testLeastSquares(self):
		testdataset = data.DataSet(self.standard_fname)
		testregression = regression.LinearRegression(testdataset)

		b1,b0 = testregression.least_squares('predictor', 'label')
		self.assertTrue(abs(b1 - 1.7) < 0.0001)
		self.assertTrue(abs(b0 - 1.9) < 0.0001)

if __name__ == '__main__':
	unittest.main()