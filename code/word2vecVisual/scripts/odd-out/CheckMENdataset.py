# Script to check the performance on MEN dataset for similarity
# using spearman correlation

import scipy as sp
import numpy as np
import scipy.stats
import sys
sys.path.append('../utils/');
from Word2Vec import loadWord2Vec, computeEmbedding

# Function to compute the correlation given the embedding path
def computeCorr(pairs, human, embedPath):
    # Load the embeddings
    (embeds, embedDim, vocabSize) = loadWord2Vec(embedPath);

    # Compute similarity for each
    simVals = [computeEmbedding(i[0], embeds, embedDim).dot(\
            computeEmbedding(i[1], embeds, embedDim)) for i in pairs];
   
    return sp.stats.spearmanr(human, simVals);
#########################################################################

# Read the pairs along with human agreement
dataPath = '/home/satwik/VisualWord2Vec/data/odd-one/MEN/';
with open(dataPath + 'MEN_dataset_lemma_form_full', 'rb') as dataFile:
    lines = [i.strip('\n').split(' ') for i in dataFile.readlines()];

# Remove the trailing POS for the words
pairs = [[j[:-2] for j in i[:-1]] for i in lines];
human = np.array([i[-1] for i in lines]);

# After: word2vec_wiki_iter_after.bin
# Before: word2vec_wiki_iter_before.bin
# Read w2v, compute correlation for wiki
modelPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/' + \
                'modelsNdata/wiki_iters/word2vec_wiki_iter_before.bin';
corr = computeCorr(pairs, human, modelPath);
print 'Correlation (before): %f' % corr[0]
modelPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/' + \
                'modelsNdata/wiki_iters/word2vec_wiki_iter_after.bin';
corr = computeCorr(pairs, human, modelPath);
print 'Correlation (after): %f' % corr[0]

# Read vis-w2v for coco, compute correlation
'''for i in xrange(0, 34):
    modelPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/' + \
                    'modelsNdata/coco_iters/word2vec_iter_%d.bin' % i;
    #                'modelsNdata/wiki_iters/word2vec_wiki_iter_before.bin';
    corr = computeCorr(pairs, human, modelPath);
    print 'Correlation (after): %f' % corr[0]'''
