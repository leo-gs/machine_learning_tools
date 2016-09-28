import sys
sys.path.append('../../')

import data
import objectives

import numpy as np
import unittest

class TestObjectives(unittest.TestCase):

	def setUp(self):
		self.mixeddata_fname = '../testdata/test_mixeddata.csv'
		self.performance_fname = '../testdata/test_performancedata.csv'
		self.entropy_fname = '../testdata/test_entropydata.csv'
		self.informationgain_fname = '../testdata/test_informationgaindata.csv'
		self.split_fname = '../testdata/test_splitdata.csv'

	def testCountLabels_integerData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		result_count = objectives.count_labels(testdataset, 'int')
		expected_count = {1:2, 7:1, -4:1}

		self.assertEqual(result_count, expected_count)

	def testCountLabels_floatData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		result_count = objectives.count_labels(testdataset, 'float')
		expected_count = {0.5:2, 0.0:1, 99.1:1}

		self.assertEqual(result_count, expected_count)

	def testCountLabels_textData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		result_count = objectives.count_labels(testdataset, 'text')
		expected_count = {'hi':2, 'hello':2}

		self.assertEqual(result_count, expected_count)

	def testCountLabels_booleanData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		result_count = objectives.count_labels(testdataset, 'boolean')
		expected_count = {True:2, False:2}

		self.assertEqual(result_count, expected_count)

	def testNoEntropy_integerData(self):
		testdataset = data.DataSet(self.entropy_fname)
		result = objectives.entropy(objectives.count_labels(testdataset, 'no_entropy'))
		expected = 0.0

		self.assertEqual(result, expected)

	def testHighEntropy_integerData(self):
		testdataset = data.DataSet(self.entropy_fname)
		result = objectives.entropy(objectives.count_labels(testdataset, 'high_entropy'))
		expected = 1.0

		self.assertEqual(result, expected)

	def testMediumEntropy_integerData(self):
		testdataset = data.DataSet(self.entropy_fname)
		result = objectives.entropy(objectives.count_labels(testdataset, 'medium_entropy'))
		expected = 0.811

		self.assertTrue(abs(result-expected) < 0.001)

	def testSplit_integerData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		splits = objectives.split_data(testdataset, 'int', 0)

		self.assertTrue(np.array_equal(splits[0].datapoints['int'], np.array([-4])))
		self.assertTrue(np.array_equal(splits[1].datapoints['int'], np.array([1, 1, 7])))

	def testSplit_floatDataNoSplit(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		splits = objectives.split_data(testdataset, 'float', -1.0)

		self.assertTrue(np.array_equal(splits[0].datapoints['float'], np.array([])))
		self.assertTrue(np.array_equal(splits[1].datapoints['float'], np.array([0.5, 0.5, 0.0, 99.1])))

	def testSplit_booleanData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		splits = objectives.split_data(testdataset, 'boolean', True)

		self.assertTrue(np.array_equal(splits[0].datapoints['boolean'], np.array([False, False])))
		self.assertTrue(np.array_equal(splits[1].datapoints['boolean'], np.array([True, True])))

	def testSplit_textData(self):
		testdataset = data.DataSet(self.mixeddata_fname)
		splits = objectives.split_data(testdataset, 'text', 'hi')

		self.assertTrue(np.array_equal(splits[0].datapoints['text'], np.array(['hello', 'hello'])))
		self.assertTrue(np.array_equal(splits[1].datapoints['text'], np.array(['hi', 'hi'])))

	def testInformationGainInt_gain(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'gain_int', 3)
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 1.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainInt_loss(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'nochange_int2', 3)
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainInt_noChange(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'nochange_int', 3)
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainInt_singleDatapoint(self):
		testdataset = objectives.split_data(data.DataSet(self.informationgain_fname), 'gain_int', 1)[0]
		split_datasets = objectives.split_data(testdataset, 'nochange_int', 3)
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainText_gain(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'gain_text', 'hi')
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 1.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainText_noChange(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'nochange_text2', 'hi')
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainText_noChange(self):
		testdataset = data.DataSet(self.informationgain_fname)
		split_datasets = objectives.split_data(testdataset, 'nochange_text', 'hi')
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testInformationGainText_singleDatapoint(self):
		testdataset = objectives.split_data(data.DataSet(self.informationgain_fname), 'gain_int', 1)[0]
		split_datasets = objectives.split_data(testdataset, 'nochange_text', 1)
		counts = objectives.count_labels(testdataset, 'label')
		testinformationgain = objectives.information_gain(counts, objectives.count_labels(split_datasets[0], 'label'), objectives.count_labels(split_datasets[1], 'label'))
		expected = 0.0

		self.assertEqual(testinformationgain, expected)

	def testFindOptimalSplit_easySplit(self):
		testdataset = data.DataSet(self.split_fname)
		testresult = objectives.find_optimal_split(testdataset, 'label', attribute='easy_split')
		expectedthreshold = 3
		expectedattribute = 'easy_split'
		expectedgain = 1.0
		expected = (expectedthreshold, expectedattribute, expectedgain)

		self.assertEqual(testresult, expected)

	def testFindOptimalSplit_noSplit(self):
		testdataset = data.DataSet(self.split_fname)
		result = objectives.find_optimal_split(testdataset, 'label', attribute='no_split')
		expectedthreshold = None
		expectedattribute = None
		expectedgain = 0
		expected = (expectedthreshold, expectedattribute, expectedgain)

		self.assertEqual(result, expected)

	def testFindOptimalSplit_mediumSplit(self):
		testdataset = data.DataSet(self.split_fname)
		resultthreshold, resultattribute, resultgain = objectives.find_optimal_split(testdataset, 'label', attribute='medium_split')
		expectedthreshold = 3
		expectedattribute = 'medium_split'
		expectedgain = 0.188
		expected = (expectedthreshold, expectedattribute, expectedgain)

		self.assertEqual(resultthreshold, expectedthreshold)
		self.assertEqual(resultattribute, expectedattribute)
		self.assertTrue(abs(resultgain-expectedgain) < 0.001)

	def testCountLabelsPerformance(self):
		testdataset = data.DataSet(self.performance_fname)
		result_count_int = objectives.count_labels(testdataset, 'int')
		result_count_float = objectives.count_labels(testdataset, 'float')
		result_count_boolean = objectives.count_labels(testdataset, 'boolean')
		result_count_text = objectives.count_labels(testdataset, 'text')
		expected_count_int = {1:276480}
		expected_count_float = {0.1:276480}
		expected_count_boolean = {True:276480}
		expected_count_text = {'text':276480}

		self.assertEqual(result_count_int, expected_count_int)
		self.assertEqual(result_count_float, expected_count_float)
		self.assertEqual(result_count_boolean, expected_count_boolean)
		self.assertEqual(result_count_text, expected_count_text)

	def testEntropyPerformance(self):
		testdataset = data.DataSet(self.performance_fname)
		entropy = objectives.entropy(objectives.count_labels(testdataset, 'int'))

	def testSplitPerformance(self):
		testdataset = data.DataSet(self.performance_fname)
		splits = objectives.split_data(testdataset, 'text', 'text')

if __name__ == '__main__':
	unittest.main()