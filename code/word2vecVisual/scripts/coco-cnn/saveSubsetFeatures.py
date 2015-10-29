# Script to save the features and tuples for the subset of COCO features
import pickle
import numpy as np
import re
import sys

# Read the split
dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/';
with open(dataPath + 'coco_subset_tuples.p', 'rb') as dataFile:
    data = pickle.load(dataFile);
#==========================================================
# Read the features and their tags
featPath = dataPath + 'train2014_fc7.npy';
featTagPath = dataPath + 'img_name_train.npy';

# Read the features and corresponding image names
features = np.load(open(featPath, 'rb'));

with open(featTagPath, 'rb') as dataFile:
    featImgNames = np.load(dataFile);

# Get the feature tags
prefix = '/srv/share/data/mscoco/coco/images/train2014/COCO_train2014_(\d*).jpg';
featTags = [str(int(re.match(prefix, i).group(1))) for i in featImgNames];

# Make a dictionary for easier access
featDict = {featTags[i]: features[i] for i in xrange(0, features.shape[0])};

#==========================================================
# For each relations, extract a tuple from corresponding image
with open(dataPath + 'coco_train_minus_cs_test_tuples.p', 'rb') as dataFile:
    tups = pickle.load(dataFile);

# Collect the tuples and corresponding features
trainData = {};
trainData['tups'] = [];
trainData['feats'] = [];
for r in data:  
    imgs = data[r];
    for i in imgs:
        imgTups = tups[i];
        # Sanity check
        if(len(imgTups) <= 0):
            print 'Empty image tuple'
            sys.error(1);

        possTup = [t for t in imgTups if t[1] == r];
        if(len(possTup) < 1):
            print 'Empty image tuple'
            sys.error(1);

        trainData['tups'].append(tuple(possTup[0]));
        trainData['feats'].append(featDict[i]);

# Check if all the tuple image are present in feature tags
'''for i in data.values():
    for j in i:
        if j not in featTags:
            print 'Key mismatch error'
            sys.error(1);'''


# Shuffle the data
noTuples = len(trainData['tups']);
shuffle = np.random.permutation([i for i in xrange(0, noTuples)]);

tups = [trainData['tups'][i] for i in shuffle];
feats = [trainData['feats'][i] for i in shuffle];

trainData['tups'] = tups;
trainData['feats'] = feats;

# Save the features in desired format along with features
pickle.dump(trainData, open(dataPath + 'coco_subset_traindata.p', 'wb'));

# Tuples in the format
# <primary:secondary:relation>
psrFile = dataPath + 'PSR_features_coco.txt';
fileId = open(psrFile, 'wb');
[fileId.write('<%s:%s:%s>\n' % (i[1], i[2], i[0])) for i in trainData['tups']];
fileId.close();

# Features in the format
# Dimension
# <features>
np.savetxt('float_features_coco.txt', np.array(trainData['feats']), delimiter = ' ', \
            fmt= '%.6f', header = str(trainData['feats'][0].shape[0]), comments = '');
