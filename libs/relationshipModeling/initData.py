"""
Filename: curateData.py

Created on : 4/19/15 4:24 PM by rama

Load the data required to test tuples for Common Sense ICCV 15 submission

"""
import os
import json


class LoadData():
    def __init__(self, val_dataset, test_dataset):
        if val_dataset == 'cs_val.json':
            self.val_path = os.path.join('myTupleExtractor', 'data', 'valDatasetCS.json')
        else:
            print "Please check the VAL filename"

        if test_dataset == 'cs_test.json':
            self.test_path = os.path.join('myTupleExtractor', 'data', 'testDatasetCS.json')
        else:
            print "Please check the TEST filename"

    def loadDataset(self, dataset):
        if dataset == 'val':
            data = json.loads(open(self.val_path).read())
        elif dataset == 'test':
            data = json.loads(open(self.test_path).read())

        tuples = []
        old_label = []
        scores = []
        for index in data:
            tuples.append(data[index][0])
            old_label.append(data[index][1])
            scores.append(data[index][2])

        return tuples, old_label, scores
