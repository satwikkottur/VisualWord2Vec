# Module containing helper functions for Word2Vec
import numpy as np
import re

# Read the word2vec embeddings
def loadWord2Vec(embedPath):
    # Reading the file
    with open(embedPath, 'rb') as dataFile:
        lines = [i.strip('\n') for i in dataFile.readlines()];

    # Read number of vocabsize and dimension
    regObj = re.match('(\d*) (\d*)', lines[0]);
    embedDim = int(regObj.group(2));
    vocabSize = int(regObj.group(1));

    embeds = {};
    for i in lines[1:]:
        iSplit = i.split(' ', 1)
        word = iSplit[0];
        embeds[word] = np.fromstring(iSplit[1], dtype = float, sep = ' ');

    print 'Done reading the embeddings!'
    return (embeds, embedDim, vocabSize)

# Compute the word2vec embeddings for a phrase given the dict
def computeEmbedding(phrase, embeds, embedDim):
    # Gather the vector embeddings
    embedWord = np.array([embeds[i.lower()] for i in phrase.split(' ') if i.lower() in embeds]);

    # None of the words are in dict
    if len(embedWord) == 0:
        embedding = np.zeros(embedDim);
    else:
        embedding = np.mean(embedWord, axis = 0);
        # Normalize and return
        embedding = embedding/np.linalg.norm(embedding);

    return embedding;
