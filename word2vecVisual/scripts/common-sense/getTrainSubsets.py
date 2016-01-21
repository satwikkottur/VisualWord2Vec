# This script creates multuple subsets of the data to get the
# mean and std of performance by vis-w2v
import re
import pdb
import random

dataPath = '/home/satwik/VisualWord2Vec/data/';

subsetSize = 10000;
# Read the test file
with open(dataPath + 'test_features.txt', 'r') as fileId:
    test = [i.strip('\n') for i in fileId.readlines()];

noFiles = 20;
savePath = dataPath + 'common-sense/test_features_subset_%02d.txt';
for i in xrange(noFiles):
    subset = random.sample(test, subsetSize);
    with open(savePath % i, 'w') as fileId:
        [fileId.write(j + '\n') for j in subset];
