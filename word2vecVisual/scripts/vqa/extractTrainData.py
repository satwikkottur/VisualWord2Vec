# Script to get the aligned tuples and features to train vis-w2v
from scipy.io import loadmat
import re
import random
import numpy as np
import os
import pdb

# Setting up the paths
dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
featPath = dataPath + 'iccv_features/';

# Read the tuple dictionary
with open(dataPath + 'vqa_tuples_train_dict.txt', 'r') as fileId:
    tuples = [re.search('<([^<>:]*):([^<>:]*):([^<>:]*)>\((\d*),(\d*)\)', i.strip('\n'))\
                            for i in fileId.readlines()];

# Shuffle the tuples
random.shuffle(tuples);

# Open a new file for tuples and features
tupleId = open(dataPath + 'vqa_psr_features.txt', 'w');

# Read the features from the folder, one by one
# and write the aligned features in the new files
features = [];
tupleSaveFmt = '<%s:%s:%s>\n';
noTuples = 0;
for tup in tuples:
    # Check if the features exists
    path = featPath + str(tup.group(4)) + '.mat';
    if not os.path.exists(path):
        continue

    # Else write the tuple and feature
    tupleId.write(tupleSaveFmt % (tup.group(1), tup.group(2), tup.group(3)));
    matData = loadmat(path);

    features.append(matData['feat'][0]);

    noTuples += 1;
    print 'Saving tuple: ' + str(noTuples)

tupleId.close();

# Save the features
featSavePath = dataPath + 'vqa_float_features.txt';
np.savetxt(featSavePath, np.array(features), fmt='%0.6f', delimiter=' ',
                    header=str(len(features[0])), comments='');
