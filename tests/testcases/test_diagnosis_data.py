import sys
sys.path.append('../../')

import data
import random_forest

import numpy as np

label = 'nephritis'

dataset = data.DataSet('../testdata/diagnosis_data.csv', ratio_validation=0.1, shuffle=True)
forest = random_forest.RandomForest(dataset, label)
forest.grow(max_depth=9, size=100)

results = forest.predict(dataset.validation_datapoints)
for point in results:
	# print point
	print str(point[label]) + ', ' + str(point['_prediction']) + ', ' + str(point['_probability'])