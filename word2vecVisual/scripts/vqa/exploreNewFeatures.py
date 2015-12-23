# Script to explore the new features for VQA dataset
import json
import pickle
import numpy as np

# All the features are arranged in the ascending order anyway
dataPath = '/home/satwik/VisualWord2Vec/data/vqa/yash/'
featPath = dataPath + 'all_scenes_train_features_parts_2.npy';
mapPath = dataPath + 'train_all_scenes_id.json';
with open(mapPath, 'rb') as dataFile:
    mapIds = json.load(dataFile);

with open(featPath, 'rb') as dataFile:
    features = np.load(dataFile);

# Save the feature
savePath = dataPath + 'float_features.txt';
noTrain = 20000;
sortedFeats = [features[mapIds.index(i)] for i in xrange(0, noTrain)];
# Write the features to the file
#np.savetxt(savePath, features, fmt = '%.6f', \
            #delimiter = ' ', header = str(features.shape[0]) + ' ' \
            #+ str(features.shape[1]), comments='');
