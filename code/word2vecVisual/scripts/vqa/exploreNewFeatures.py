# Script to explore the new features for VQA dataset
import json
import pickle
import numpy as np

# All the features are arranged in the ascending order anyway
featPath = '/srv/share/vqa/release_data/abstract_v002/scene_json/features_v2/features/';
#with open(featPath + 'abstract_v002_instances-random_files.json', 'rb')\
#                                        as dataFile:
#    mapFile = json.load(dataFile);

with open(featPath + 'abstract_v002_instances-random_features.npy', 'rb') as dataFile:
    features = np.load(dataFile);

# Choose just the starting 200 as they are the training feature
:
    
