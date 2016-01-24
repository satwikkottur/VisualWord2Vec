# This script reads the out files of runs on multiple subsets and computes std and mean
import re
import pdb
import numpy as np

dataPath = '/home/satwik/VisualWord2Vec/word2vecVisual/dumps/';

with open(dataPath + 'out_single_subset', 'r') as fileId:
#with open(dataPath + 'out_single_subset', 'r') as fileId:
    lines = [i.strip('\n') for i in fileId.readlines() if i[0:9] == 'Precision'];

# Check for the last 20 lines starting with 'Precision'
precList = [];
for line in lines[::-1][0:20]:
    match = re.match('Precision \(val, test\) : ([.\d]*) ([.\d]*)', line);
    precList.append(float(match.group(2)));

print 'Mean: %f\nStd: %f' % (np.mean(np.array(precList)), \
                                np.std(np.array(precList)))
