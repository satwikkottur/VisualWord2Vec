# Script to order the captions based on the mapping given
import pdb
from collections import defaultdict

dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';

# Open the captions
with open(dataPath + 'vqa_train_captions_lemma.txt', 'r') as fileId:
    captions = [i.strip('\n') for i in fileId.readlines()];

# Open the mapping file
with open(dataPath + 'vqa_feature_map.txt', 'r') as fileId:
    maps = [int(i.strip('\n')) for i in fileId.readlines()];

ordered = defaultdict(list);
for i in xrange(len(maps)):
    ordered[maps[i]].append(captions[i]);

# Write it back to a file
with open(dataPath + 'vqa_train_captions_lemma_order.txt', 'w') as fileId:
    [fileId.write(ordered[i][j]+ '\n') for i in xrange(len(ordered)) \
                                for j in xrange(len(ordered[1]))];
