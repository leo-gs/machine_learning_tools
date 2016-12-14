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

		testregression.least_squares('predictor', 'label')
		self.assertEqual(int(testregression.b1), 1)
		self.assertEqual(int(testregression.b0), 0)

	def testLeastSquares(self):
		testdataset = data.DataSet(self.standard_fname)
		testregression = regression.LinearRegression(testdataset)

		testregression.least_squares('predictor', 'label')
		self.assertTrue(abs(testregression.b1 - 1.7) < 0.0001)
		self.assertTrue(abs(testregression.b0 - 1.9) < 0.0001)

	def testPrediction_easy(self):
		testdataset = data.DataSet(self.easy_fname)
		testregression = regression.LinearRegression(testdataset)

		testregression.least_squares('predictor', 'label')
		prediction = testregression.predict(1)
		self.assertEqual(int(prediction), 1)

	def testPrediction(self):
		testdataset = data.DataSet(self.standard_fname)
		testregression = regression.LinearRegression(testdataset)

		testregression.least_squares('predictor', 'label')
		prediction = testregression.predict(0)
		self.assertTrue(abs(prediction - 1.9) < 0.0001)

if __name__ == '__main__':
	unittest.main()