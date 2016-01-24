# Script to read the tokens
import pdb

dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';

rawPath = dataPath + 'coco_vqa_train_captions_lemma.txt';

with open(rawPath, 'r') as fileId:
    lines = [i.strip('\n').strip('.') for i in fileId.readlines()];

tokens = [];
for i in lines:
    tokens.extend(i.split(' '));
print len(set(tokens))
print len(tokens)
pdb.set_trace();
