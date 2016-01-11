# Script to explore the datasets and splitting the features corresponding
import pdb
import os

dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';

# Get the list of the images
listing = os.listdir(dataPath + 'images/')

# Read the features file
featPath = dataPath + 'fc7_train_vectors.txt';
with open(featPath, 'r') as fileId:
    line = [int(i.split('\t')[0]) for i in fileId.readlines()];

pdb.set_trace();

