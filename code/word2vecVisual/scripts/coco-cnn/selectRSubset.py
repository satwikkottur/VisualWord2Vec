# Prepare the training data for MS COCO that matches with ICCV 
# abstract dataset
import numpy as np
import pickle
import json
import re
import sys

dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/';
tuplesPath = dataPath + 'coco_train_minus_cs_test_tuples.p';

with open(tuplesPath, 'rb') as dataFile:
    cocoTups = pickle.load(dataFile);
#==========================================================
# Read the captions
capPath = dataPath + 'coco_train_minus_cs_test.json';
with open(capPath, 'rb') as dataFile:
    capData = json.load(dataFile);

#==========================================================

# Get a count of number of different images each relation is present
rCount = {};
rIndices = {};
for i in cocoTups:
    # Check for each relation and different image
    for j in cocoTups[i]:
        rel = j[1];
        # Check for count and list
        if rel in rCount:
            rCount[rel] += 1;
            if i not in rIndices[rel]:
                rIndices[rel].append(i)
        else:
            rCount[rel] = 1;
            rIndices[rel] = [];

rListCount = [len(rIndices[i]) for i in rIndices];
rListCount.sort(reverse=True);

noR = 213;
threshold = rListCount[noR];

# Select any tuple that has rListCount greater than threshold
# Set of images collected till now
imgSet = set([]);
R = [];
rImgMap = {};

for i in rIndices:
    # Threshold
    if(len(rIndices[i]) >= threshold):
        # Check if difference is greater than 20
        diff = set(rIndices[i]) - imgSet;
        # Add if >= 20
        if(len(diff) >= 20):
            if(i in R):
                print 'Some error!'
                sys.exit(1);
            R.append(i);
            rImgMap[i] = list(diff)[0:20];
            imgSet.update(rImgMap[i])

    # Break if number of R reaches 213
    if(len(R) == 213):
        break;
        
# Save this as a pickle
pickle.dump(rImgMap, open(dataPath + 'coco_subset_tuples.p', 'wb'));

# Read the features
'''featPath = dataPath + 'train2014_fc7.npy';
featTagPath = dataPath + 'img_name_train.npy';

# Read the features and corresponding image names
features = np.load(open(featPath, 'rb'));

with open(featTagPath, 'rb') as dataFile:
    featImgNames = np.load(dataFile);

#==========================================================
# Get the feature tags
prefix = '/srv/share/data/mscoco/coco/images/train2014/COCO_train2014_(\d*).jpg';
featTags = [str(int(re.match(prefix, i).group(1))) for i in featImgNames];

# Collect the relevant features and captions
featDumpPath = dataPath + 'fc7_features_train.txt';
capDumpPath = dataPath + 'captions_coco_train.txt';
mapDumpPath = dataPath + 'captions_coco_train_map.txt';

# Open the files
#capId = open(capDumpPath, 'wb');
mapId = open(mapDumpPath, 'wb');

# Get the corresponding feature for each caption
# Also save the map
trainFeats = [];
count = 0;
for i in capData:
    if( count % 1000 == 0):
        print '%d / %d' % (count, len(capData))
    count += 1;

    # Get feat index
    featId = featTags.index(i);
    
    # Cross check
    if featTags[featId] != i:
        sys.exit(1);

    # For each caption, note feat id and write the map
    #[capId.write( '%d: %s' % (len(trainFeats), j.replace('\n', '')) + '\n' )\
    #                                for j in capData[i]];
    [mapId.write('%d\n' % len(trainFeats)) for j in capData[i]];

    # Collect feature
    trainFeats.append(features[featId]);

#capId.close();
mapId.close();

print 'Preparing to dump feature file'
# Dump all the features
#np.savetxt(featDumpPath, np.array(trainFeats), delimiter = ' ', \
#                                         fmt = '%.6f', header = '%d %d' % (len(trainFeats), trainFeats[0].shape[0]));
print 'Finished dumping feature file'''
