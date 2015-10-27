# Script to prepare captions and features for CNN features of COCO
import json
import numpy as np
#from scipy.cluster.vq import vq, kmeans, whiten
from sklearn.cluster import KMeans
import re

dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/'
captionPath = dataPath + 'captions_train2014.json';

with open(captionPath, 'rb') as dataFile:
    data = json.load(dataFile);

featPath = dataPath + 'train2014_fc7.npy';
featTagPath = dataPath + 'img_name_train.npy';

# Read the features and cluster them through k-means
features = np.load(open(featPath, 'rb'));
kmeans = KMeans(n_clusters = 100);
kmeans.fit(features);
clusterId = kmeans.labels_;
print 'Done clustering!'

featPaths = np.load(open(featTagPath, 'rb'));

# Get the feature tags
prefix = '/srv/share/data/mscoco/coco/images/train2014/';
featTags = [i.split(prefix)[1] for i in featPaths];

# Get the corresponding feature for each caption
# Also save the map
captionFeatMap = [];
for i in data['annotations']:
    capImgId = i['image_id'];

    # Create the image tag
    imgTag = 'COCO_train2014_%012d.jpg' % capImgId;
    # Get the corresponding feature index
    captionFeatMap.append(featTags.index(imgTag));

# Write the features into a file
#featDumpPath = dataPath + 'features_coco.txt';
#featId = open(featDumpPath, 'wb');

# Write the captions into a file (to lemmatize later)
'''capDumpPath = dataPath + 'captions_coco_train.txt';
capId = open(capDumpPath, 'wb');

# Replace all \n with ''
for i in data['annotations']:
    capId.write(i['caption'].replace('\n', '') + '\n');

capId.close();'''

# Write the clusterIds to a file
# We take the captionFeatMap to get correct id
clusterPath = dataPath + 'cluster_100_coco_train_nohup.txt';
cId = open(clusterPath, 'wb');

for i in xrange(0, len(data['annotations'])):
    # Get corresponding id
    clustId = clusterId[captionFeatMap[i]];

    cId.write(str(clustId) + '\n');

cId.close();
