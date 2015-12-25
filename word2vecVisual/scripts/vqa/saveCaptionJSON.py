# Script to save the lemmantized captions in JSON format for tuple extractor
import json

dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
with open(dataPath + 'vqa_train_captions_lemma.txt', 'rb') as dataFile:
    captions = [i.strip('\n') for i in dataFile.readlines()];

with open(dataPath + 'vqa_feature_map.txt', 'rb') as dataFile:
    imgMap = [i.strip('\n') for i in dataFile.readlines()];

# Populating the dictionary
imgCaps = {};
for i in xrange(0, len(captions)):
    if(imgMap[i] in imgCaps):
        imgCaps[imgMap[i]].append(captions[i]); 
    else:
        imgCaps[imgMap[i]] = [captions[i]];

# Saving as json file
with open(dataPath + 'vqa_train_captions_lemma.json', 'wb') as dataFile:
    json.dump(imgCaps, dataFile);
    
    
    
