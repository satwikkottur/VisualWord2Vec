# Script to save the features for vqa
import numpy as np

dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
features = np.load(dataPath + 'abstract_v002_train2015_features.npy');

# Analyse the sparsity
nz = [np.count_nonzero(i) for i in features[0:20]];

# Brute force dump
np.savetxt(dataPath + 'float_features_vqa.txt', features, fmt = '%.6f', \
                    delimiter = ' ', header = str(features.shape[1]), comments='');
