# Script to prepare the tuples from MS COCO
import pickle

dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/';
tuplePath = dataPath + 'coco_train_minus_cs_test_tuples.p';

with open(tuplePath, 'rb') as dataFile:
    data = pickle.load(dataFile);

data
