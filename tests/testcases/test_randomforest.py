import sys
sys.path.append('../../')

import data
import objectives
import random_forest

import numpy as np
import unittest

class TestRandomForest(unittest.TestCase):

	def setUp(self):
		self.mixeddata_fname = '../testdata/test_mixeddata.csv'
		self.performance_fname = '../testdata/test_performancedata.csv'
		self.randomforest_fname = '../testdata/test_randomforestdata.csv'
		self.randomforest2_fname = '../testdata/test_randomforestdata2.csv'
		self.randomforestperformance_fname = '../testdata/test_randomforestperformance.csv'

	def testGrowTree_oneSplit(self):
		testdataset = data.DataSet(self.randomforest_fname)
		testtree = random_forest.Tree(testdataset, max_depth=1, label='label', m=2)

		self.assertEqual(testtree.root.left_node.left_node, None)
		self.assertEqual(testtree.root.right_node.right_node, None)
		self.assertEqual(testtree.root.left_node.probabilities.values()[0], 1.0)
		self.assertEqual(len(testtree.root.left_node.probabilities), 1)
		self.assertEqual(testtree.root.right_node.probabilities.values()[0], 0.5)
		self.assertEqual(testtree.root.right_node.probabilities.values()[1], 0.5)
		self.assertEqual(len(testtree.root.right_node.probabilities), 2)

	def testGrowTree_twoSplits(self):
		testdataset = data.DataSet(self.randomforest2_fname)
		testtree = random_forest.Tree(testdataset, max_depth=2, label='label', m=2)

		probabilities = [
				testtree.root.left_node.left_node.probabilities,
				testtree.root.left_node.right_node.probabilities,
				testtree.root.right_node.left_node.probabilities,
				testtree.root.right_node.right_node.probabilities
		]
		for probability in probabilities:
			self.assertEqual(probability.values()[0], 1.0)
			self.assertEqual(len(probability), 1)

	def testTreePredict(self):
		testdataset = data.DataSet(self.randomforest2_fname, ratio_validation=0.25, shuffle=True)
		testtree = random_forest.Tree(testdataset, max_depth=2, label='label', m=2)
		for datapoint in testdataset.validation_datapoints:
			dataset = data.DataSet(datapoints=datapoint)
			probabilities = testtree.get_probabilities(datapoint)

			# results are too random for asserts

	def testInitForest(self):
		testdataset = data.DataSet(self.randomforest2_fname, ratio_validation=0.25, shuffle=True)
		testforest = random_forest.RandomForest(testdataset, 'label')

		self.assertEqual(testforest.dataset, testdataset)
		self.assertEqual(testforest.label, 'label')

	def testGrowForest_size1(self):
		testdataset = data.DataSet(self.randomforest2_fname, ratio_validation=0.25, shuffle=True)
		testforest = random_forest.RandomForest(testdataset, 'label')
		testforest.grow(max_depth=2, m=2)
		predictions = testforest.predict(testdataset.validation_datapoints)

		# results are too random for asserts

	def testGrowForest_size100(self):
		testdataset = data.DataSet(self.randomforest2_fname, ratio_validation=0.25, shuffle=True)
		testforest = random_forest.RandomForest(testdataset, 'label')
		testforest.grow(size=100, max_depth=2, m=2)
		predictions = testforest.predict(testdataset.validation_datapoints)

		# results are too random for asserts

	# def testGrowForestPerformance(self):
	# 	testdataset = data.DataSet(self.randomforestperformance_fname, ratio_validation=0.25, shuffle=True)
	# 	testforest = random_forest.RandomForest(testdataset, 'text')
	# 	testforest.grow(size=1000, max_depth=2)
	# 	predictions = testforest.predict(testdataset.validation_datapoints)

	# 	for datapoint in predictions:
	# 		self.assertEqual(datapoint['label'], datapoint['prediction'])

if __name__ == '__main__':
	unittest.main()