# This script allows for computing the featurs for sentences using word2vec
import numpy as np
import sys
import pdb

# adding the path to reading word2vec
sys.path.append('../utils/');
from Word2Vec import *

# Read embeddings
dataPath = '/home/satwik/VisualWord2Vec/data/';

# MSCOCO before
#embedPath = dataPath + 'al_vectors.txt';
# MSCOCO after
#embedPath = dataPath + 'word2vec_output_bestmodel_single.bin';
# Wiki before
embedPath = dataPath + 'word2vec_wiki_iter_before.bin';
# Wiki after
#embedPath = dataPath + 'word2vec_wiki_iter_after.bin';
(embeds, embedDim, vocabSize) = loadWord2Vec(embedPath);

# Read all the coco captions and write training and testing separately
cocoPath = dataPath + 'coco-cnn/captions_coco_train.txt'
with open(cocoPath, 'r') as fileId:
    lines = [line.strip('\n').split(': ') for line in fileId.readlines()];

# Get the embeddings for the sentences
# features = [computeEmbedding(line[1], embeds, embedDim) for line in lines];

# Opening the files to save the embeddings
trainPath = dataPath + 'coco-cca/wiki/train_caption_embeds_before.txt';
testPath = dataPath + 'coco-cca/wiki/test_caption_embeds_before.txt';
mapPath = dataPath + 'coco-cca/wiki/test_caption_maps.txt';

testFeats = [];
trainFeats = [];
mapId = open(mapPath, 'w');

seenImgs = set([]);
lineId = 0;
for line in lines:
    # Already seen, write in test and map
    imgId = int(line[0]);
    if imgId in seenImgs:
        testFeats.append(computeEmbedding(line[1], embeds, embedDim));
        mapId.write(line[0] + '\n');
    
    # First time, write in train and add to seenImgs
    else:
        trainFeats.append(computeEmbedding(line[1], embeds, embedDim));
        seenImgs.add(imgId);

    # Increment the id for debug
    if lineId % 5000 == 0:
        print 'Current line : %d ...' % lineId;
    lineId +=1;

mapId.close();

# Use numpy to write down the features
np.savetxt(trainPath, np.array(trainFeats), delimiter=' ', fmt='%.6f', \
                header='%d %d' % (len(trainFeats), embedDim), comments='');
np.savetxt(testPath, np.array(testFeats), delimiter=' ', fmt='%.6f', \
                header='%d %d' % (len(testFeats), embedDim), comments='');
