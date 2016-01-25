# This script allows for computing the featurs for sentences using word2vec
import numpy as np
import sys
import pdb
import random

# adding the path to reading word2vec
sys.path.append('../utils/');
from Word2Vec import *

# Read embeddings
dataPath = '/home/satwik/VisualWord2Vec/data/';

# Read all the coco captions and write training and testing separately
cocoPath = dataPath + 'coco-cnn/captions_coco_train.txt'
with open(cocoPath, 'r') as fileId:
    lines = [line.strip('\n').split(': ') for line in fileId.readlines()];

# Opening the files to save the embeddings
valPath = dataPath + 'coco-cnn/captions_coco_val_nomaps.txt';
testPath = dataPath + 'coco-cnn/captions_coco_test_nomaps.txt';
gtPath = dataPath + 'coco-cnn/captions_coco_val_gtruth.txt';
gtPathTest = dataPath + 'coco-cnn/captions_coco_test_gtruth.txt';
trainPath = dataPath + 'coco-cnn/captions_coco_dataset_nomaps.txt'

valId = open(valPath, 'w');
gtId = open(gtPath, 'w');
trainId = open(trainPath, 'w');
testId = open(testPath, 'w');
gtIdTest = open(gtPathTest, 'w');

seenImgs = set([]);
lineId = 0;
for line in lines:
    # Already seen, write in test and map
    imgId = int(line[0]);
    if imgId in seenImgs:
        # Select the image to be in validation set based on some random number
        if random.random() < 0.05:
            valId.write(line[1] + '\n');
            gtId.write(line[0] + '\n');
        else:
            testId.write(line[1] + '\n');
            gtIdTest.write(line[0] + '\n');
    
    # First time, write in train and add to seenImgs
    else:
        trainId.write(line[1] + '\n');
        seenImgs.add(imgId);

    # Increment the id for debug
    if lineId % 5000 == 0:
        print 'Current line : %d ...' % lineId;
    lineId +=1;

valId.close();
gtId.close();
trainId.close();
testId.close();
gtIdTest.close();
