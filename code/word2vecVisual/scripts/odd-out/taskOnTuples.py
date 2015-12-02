# Script to generate the pilot for odd one out
import random
from OddFinder import OddFinder
import re

# Read the tuples file
dataPath = '/home/satwik/VisualWord2Vec/data/odd-one/vis-w2v-tuples.txt';

with open(dataPath, 'rb') as dataFile:
    lines = [re.match('([^:|]*):([^:|]*):([^:|]*).*\n', \
                        i) for i in dataFile.readlines()];

# Pick only the unique triplets 
data = [[i.group(1), i.group(2), i.group(3)] for i in lines];
[i.sort() for i in data];
data = list(set([tuple(i) for i in data]));

modelPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/';
w2vPath = modelPath + 'al_vectors.txt';
visPath = modelPath + 'word2vec_after_refine_bestmodel.bin';

finder = OddFinder();
finder.readEmbeddings(visPath, w2vPath);
finder.performTask(data);
