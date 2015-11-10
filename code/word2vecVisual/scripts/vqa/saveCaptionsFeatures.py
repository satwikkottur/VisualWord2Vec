# Script to read captions and visual features for VQA dataset
import json
import numpy as np
import io

#################################################################################
# Setting up save paths
dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
featSavePath = dataPath + 'vqa_train_features.txt';
capSavePath = dataPath + 'vqa_train_captions_raw.txt';
mapSavePath = dataPath + 'vqa_feature_map.txt';
#################################################################################
# Read the val data
#fileName = '/home/satwik/VisualWord2Vec/data/captions_abstract_v002_val2015.json';
#with open(fileName) as data_file:
#    valData = json.load(data_file);

# Read the train data
fileName = dataPath + 'captions_abstract_v002_train2015.json';
with open(fileName) as data_file:
    trainData = json.load(data_file);

# Write the map file for the captions
mapId = open(mapSavePath, 'wb');
ids = [i['image_id'] for i in trainData['annotations']];
[mapId.write(str(i)+'\n') for i in ids];
mapId.close();

# Read the features
fileName = dataPath + 'abstract_v002_instances-random_features.npy';
with open(fileName, 'rb') as dataFile:
    features = np.load(dataFile);

# Brute force dump
np.savetxt(dataPath + 'float_features_vqa.txt', features[0:19999], fmt = '%.6f', \
                    delimiter = ' ', header = str(features.shape[1]), comments='');
                    
'''# Read the features
featName = '/srv/share/vqa/release_data/abstract_v002/scene_json/features/abstract_v002_%s2015_features.npy';
trainFeat = numpy.load(featName % 'train');
#valFeat = numpy.load(featName % 'val');

# open the file
captionId = io.open(capSavePath, 'w', encoding='utf8');

# Write the sentences back to file
for i in trainData['annotations']:
    captionId.write(i['caption'] + '\n');
captionId.close();

# Write the features back to a text file (for refining code)
#featDim = len(trainFeat[0]);
#featureId = open(featSavePath, 'wb');

#featureId.write(num2str(featDim) + '\n');
#for i in 

#featureId.close()'''
