# Script to read and dump numpy feature files into txt format for
# faster clustering in C
import numpy as np
import pickle

dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/'
featPath = dataPath + 'train2014_fc7.npy';

# Read the features and cluster them through k-means
features = np.load(open(featPath, 'rb'));

np.savetxt('fc7_features.txt', features, delimiter = ' ', fmt = '%.6f',\
                        header = str(features.shape[1]));
