# Script to tokenize the data for training word2vec data
from ptbtokenizer import PTBTokenizer
import os
import json

rootPath = '/home/satwik/VisualWord2Vec';
fileName = 'coco_train_minus_cs_test.json';
filePath = os.path.join(rootPath, 'data', fileName);

tokenizer = PTBTokenizer();
jsonFile = json.load(open(filePath));

tokens = tokenizer.tokenize(jsonFile)
