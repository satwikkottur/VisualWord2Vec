# Script to tokenize the data for training word2vec data
# Takes as input a raw json file and tokenizes them creating a new json file
import os
import json
import sys
sys.path.append('/home/satwik/VisualWord2Vec/libs/');
import ptbtokenizer as pt

# Setting up the input file and dump file paths
rootPath = '/home/satwik/VisualWord2Vec';
fileName = 'coco_train_minus_cs_test.json';
inputPath = os.path.join(rootPath, 'data', fileName);
dumpPath = os.path.join(rootPath, 'data', fileName.replace('.json', '_tokenized.json'));

# Tokenizing using the PTBTokenizer (check : libs/ptbtokenizer/)
tokenizer = pt.PTBTokenizer();
jsonFile = json.load(open(inputPath));
tokens = tokenizer.tokenize(jsonFile);

# Saving the tokenized sentences as another json file
dumpFile = open(dumpPath, 'w');
json.dump(tokens, dumpFile);

close(dumpFile);
