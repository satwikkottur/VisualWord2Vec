# Script to read the cliparts, sentences and then align based on mutual 
# information
import cPickle as pickle
import numpy as np
import pdb
import itertools as it
from collections import defaultdict
import math
from copy import deepcopy

# Setup:
#   1. Type of scene for each of the training scenes
#           - pickle file with scene index : scene type (0 / 1)
#   2. List of cliparts present in the each of the training scenes
#           - pickle file with scene index : list of cliparts
#   3. Captions for each scene
#           - txt file with each caption in one line (5 per scene in order)
#   4. Image map between the captions and scenes (depleted) - always use ordered captions
# 
# Test Input:
#   1. Tuples(P, S) from the caption sentences (to align)
#
# Test Output:
#   1. Alignment between the clipart and the words

class Aligner:
    # Minimum number of times a word must occur
    minOccurance = 2;
    # Read the scene types, scene clipart and captions
    def __init__(self, sceneTypePath, clipartPath, captionPath):
        with open(sceneTypePath, 'r') as fileId:
            self.types = pickle.load(fileId);
        with open(clipartPath, 'r') as fileId:
            self.cliparts = pickle.load(fileId);
        with open(captionPath, 'r') as fileId:
            self.captions = [l.strip('\n') for l in fileId.readlines()];
            self.capWords = [l.split(' ') for l in self.captions];

    # Collecting unique clipart objects and tuples, getting counts
    def computeCounts(self):
        # Count (co)-occurance for each caption and clipart (based on scene)
        # nW[x] : word occurance
        # nC[x] : clipart occurance
        # nWC[x] : word-clipart co-occurance
        self.nW ={0:defaultdict(int), 1:defaultdict(int)};
        self.nC ={0:defaultdict(int), 1:defaultdict(int)};
        self.nWC ={0:defaultdict(int), 1:defaultdict(int)};

        # Weed out the words that dont occur more than some time
        for capId in xrange(0, len(self.capWords)):
            sceneId = int(capId/5);
            sceneType = self.types[sceneId];
            # Register the clipart
            self.increaseCount(self.nC[sceneType], self.cliparts[sceneId]);
            # Register the word in the caption
            self.increaseCount(self.nW[sceneType], self.capWords[capId]);
            # Register co-occurances
            for pair in it.product(self.capWords[capId], self.cliparts[sceneId]):
                self.increaseCount(self.nWC[sceneType], pair);

        # Remove all the words that dont occur min number of times, for both the scenes
        for s in [0, 1]:
            cutOff = [i for i in self.nW[s] if self.nW[s][i] < self.minOccurance];
            [self.nW[s].pop(i, None) for i in cutOff];
            [self.nWC[s].pop((i, c), None) for i in cutOff for c in self.nC[s]];

    # Normalize counts to get probabilities
    def normalizeCounts(self):
        for s in [0, 1]: # For both the scene types
            # Normalize 
            self.normalize(self.nW[s]);
            self.normalize(self.nC[s]);
            self.normalize(self.nWC[s]);

    # Normalize a count
    def normalize(self, counts):
        sumTotal = sum(counts.values());
        for i in counts:
            counts[i] = float(counts[i])/sumTotal;

    # Compute the alignment between words and cliparts
    def computeMI(self):
        # First compute counts
        self.computeCounts();
        # Normalize to get probabilities
        self.normalizeCounts();

        self.mi = {0:defaultdict(float), 1:defaultdict(float)}
        # Compute mutual information for (w, c) pairs
        for s in [0, 1]:
            for (w, c) in self.nWC[s]:
                self.mi[s][(w, c)] = self.nWC[s][(w, c)] * \
                                        math.log(self.nWC[s][(w, c)]/\
                                        (self.nW[s][w] * self.nC[s][c]));

    # Get the alignment for all the train tuples at once, a wrapper for alignClipart
    # Input:
    #   tuples : dictionary of caption id : tuples
    def getTupleAlignment(self, tuples):
        # Scenes that have tuples
        nonZeroId = [i for i in tuples if len(tuples[i]) > 0];

        # Get everything under one hood
        # Scenes, captions, cliparts, tuples, alignment
        alignment = {};
        for i in xrange(len(tuples)):
            alignment[i] = defaultdict(list);
        #alignment = defaultdict(lambda : defaultdict(list));
        for capId in nonZeroId:
            sceneId = capId/5;
            captionData = {'caption':self.captions[capId], \
                            'tuples':[]};

            for tup in tuples[capId]:
                (clipP, clipS) = self.alignClipart(tup[0::2], \
                            self.cliparts[sceneId], self.types[sceneId]);
                captionData['tuples'].append({'tuple':tup, 'P':clipP, 'S':clipS});

            alignment[sceneId]['captions'].append(deepcopy(captionData));
            alignment[sceneId]['clipart'] = self.cliparts[sceneId];
        return alignment

    # Compute the alignment for a (list of) word, given the word, 
    # sceneType and cliparts in the scene
    def alignClipart(self, word, cliparts, sceneType):
        if(isinstance(word, list)):
            bestAlign = tuple([self.alignClipart(w, cliparts,sceneType) \
                                                        for w in word]);
        else:
            # Handle phrases as well
            matches = [];
            scores = [];
            for part in word.split(' '):
                mi = np.array(self.getCount(self.mi[sceneType], \
                                        [(part, c) for c in cliparts]));
                best = np.argmax(mi);
                matches.append(cliparts[best]);
                scores.append(mi[best]);

            # Predict one alignment for all the phrases based on the mutual information
            bestAlign = matches[scores.index(max(scores))];

        return bestAlign

    # Handy function to increment counter for a given list / element
    # checks if a key exists else inserts otherwise
    def increaseCount(self, counter, keyList):
        # Handle lists and single values separately
        if(isinstance(keyList, list)):
            for key in keyList:
                counter[key] += 1;
        else:
            counter[keyList] += 1;

    # Handy function to check for a given list / element in counts
    # checks if a key exists, returns value if yes or 0 otherwise
    def getCount(self, counter, keyList):
        # Handle lists and single values separately
        if(isinstance(keyList, list)):
            return [counter[key] for key in keyList];
        else:
            return counter[keyList]
#**********************************************************************
# Main function showing the usage of the class Aligner
if __name__ == '__main__':
    # Add data path
    dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
    sceneTypePath = dataPath + 'scene_type.cPickle';
    clipartPath = dataPath + 'clipart_occurance.cPickle';
    captionPath = dataPath + 'vqa_train_captions_lemma_order.txt';

    #align = Aligner(sceneTypePath, clipartPath, captionPath);

    ## Compute mutual information between words and clipart
    #align.computeMI();

    # Saving the pickle for align
    savePath = dataPath + 'vqa_captions_mi.pickle';
    #with open(savePath, 'w') as fileId:
    #    pickle.dump(align, fileId);
    #print 'Saved at : %s' % savePath

    # Load the pickle for align
    with open(savePath, 'r') as fileId:
        align = pickle.load(fileId);
    print 'Loaded model from: %s' % savePath

    tuplesPath = dataPath + 'vqa_train_tuples.pickle';
    with open(tuplesPath, 'r') as fileId:
        tuples = pickle.load(fileId);

    # Get the alignments for the training tuples
    alignment = align.getTupleAlignment(tuples);

    # Save the alignment
    alignPath = dataPath + 'vqa_train_alignment.pickle';
    with open(alignPath, 'w') as fileId:
        pickle.dump(alignment, fileId);
    print 'Saved the alignment : %s' % alignPath

    # For a given word and set of cliparts, get the alignment
    #align.alignClipart(word, cliparts, sceneType);

###################################################################################
# Collection Bin (for extra code)
# Code for counting from tuples

# Register the word
#for i in self.tuples[capId]:
#    # Only primary and secondary
#    self.increaseCount(self.nW1, i[0::2])

#    # Register co-occurances
#    for (w, c) in it.product(i[0::2], self.cliparts[sceneId]):
#        self.increaseCount(self.nWC1, (w, c));

###################################################################################
# Unique clipart (scene0, scene1) and words (P, S) from tuples
#self.art1 = [];
#self.art0 = [];
#for i in self.cliparts:
#    if self.types[i]:
#        self.art1.extend(self.cliparts[i]);
#    else:
#        self.art0.extend(self.cliparts[i]);
#self.art1 = set(self.art1);
#self.art0 = set(self.art0);
#
#self.word = [k for i in self.tuples.values() for j in i \
#                                        for k in (j[0], j[2])];
#self.word = set(self.word);

###################################################################################
#artList = self.nC0.keys();
#for w in self.nW0:
#    mi = np.array(self.getCount(self.mi0, [(w, c) for c in artList]));
#    bestC = np.argmax(mi);
#    self.align0[w] = artList[bestC];
#
#artList = self.nC1.keys();
#for w in self.nW1:
#    mi = np.array(self.getCount(self.mi1, [(w, c) for c in artList]));
#    bestC = np.argmax(mi);
#    self.align1[w] = artList[bestC];
###################################################################################
