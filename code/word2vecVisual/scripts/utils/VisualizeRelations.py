# Script to visualize the relations
import re
import numpy as np
import sys
from Word2Vec import loadWord2Vec, computeEmbedding
from tsne import tsne
import matplotlib.pyplot as plot
import matplotlib
import pickle

# Read the lemmatized P, S, R relations
dataPath = '/home/satwik/VisualWord2Vec/data/';
with open(dataPath + 'PSR_features_lemma.txt', 'rb') as dataFile:
    prs = [i.strip('\n').lower() for i in dataFile.readlines()];

# Extract unique relations
relations = list(set([re.match('<[^<>:]*:[^<>]*:([^<>]*)>', i).group(1) for i in prs]));
embedR = {};

# Now read word2vec embeddings for wikipedia
wikiPath = dataPath + '../code/word2vecVisual/modelsNdata/word2vec_wiki_iter_%d.bin';
cocoPath = dataPath + '../code/word2vecVisual/modelsNdata/word2vec_iter_%d.bin';

before = loadWord2Vec(wikiPath % 0);
embedR = {i: computeEmbedding(i, before[0], before[1]) for i in relations};
# Perform t-sne
perplexity = 5;
initDims = 50;
#def tsne(X = Math.array([]), no_dims = 2, initial_dims = 50, perplexity = 30.0):
tsneBefore = tsne(np.array([embedR[i] for i in relations]), initial_dims = initDims, \
                                            perplexity = perplexity);
#plot.figure(1);
#plot.scatter(tsneBefore[:, 0], tsneBefore[:, 1], color = 'blue');
#matplotlib.rcParams.update({'font.size': 8});

# Add text to the scatter points
#for i in xrange(0, len(relations)):
    # Place them alternatively
    #plot.annotate(relations[i], (tsneBefore[i, 0] + 0.5, tsneBefore[i, 1] + 0.5));

after = loadWord2Vec(wikiPath % 24);
embedRAfter = {i: computeEmbedding(i, after[0], after[1]) for i in relations};
tsneAfter = tsne(np.array([embedRAfter[i] for i in relations]), initial_dims = initDims, \
                                                        perplexity = perplexity);
#plot.figure(2);
#plot.scatter(tsneAfter[:, 0], tsneAfter[:, 1], color = 'red');
# Add text to the scatter points
#for i in xrange(0, len(relations)):
    # Place them alternatively
    #plot.annotate(relations[i], (tsneAfter[i, 0] + 0.5, tsneAfter[i, 1] + 0.5));


# Save the embeddings for local display
data = {};
data['relations'] = relations;
data['tsneBefore'] = tsneBefore;
data['tsneAfter'] = tsneAfter;

with open('tsne_pickledump.pickle', 'wb') as dataFile:
    pickle.dump(data, dataFile);

#plot.show()
