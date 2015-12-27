# Script to read the cliparts, sentences and then align based on mutual 
# information
import cPickle as pickle
import numpy as np
import pdb
import itertools as it
from collections import defaultdict
import math

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
            self.capWords = [l.strip('\n').split(' ') for l in fileId.readlines()];
            
    # Collecting unique clipart objects and tuples, getting counts
    def computeCounts(self):
        # Count (co)-occurance for each caption and clipart (based on scene)
        # nWx : word occurance
        # nCx : clipart occurance
        # nWCx : word-clipart co-occurance
        self.nW0 = defaultdict(int); 
        self.nC0 = defaultdict(int); 
        self.nWC0 = defaultdict(int);
        self.nW1 = defaultdict(int);  
        self.nC1 = defaultdict(int);  
        self.nWC1 = defaultdict(int); 

        # Weed out the words that dont occur more than some time
        for capId in xrange(0, len(self.capWords)):
            sceneId = int(capId/5);
            if self.types[sceneId]:
                # Register the clipart
                self.increaseCount(self.nC1, self.cliparts[sceneId]);
                # Register the word in the caption
                self.increaseCount(self.nW1, self.capWords[capId]);
                # Register co-occurances
                for pair in it.product(self.capWords[capId], self.cliparts[sceneId]):
                    self.increaseCount(self.nWC1, pair);

            else:
                # Register the clipart
                self.increaseCount(self.nC0, self.cliparts[sceneId])
                # Register the word
                self.increaseCount(self.nW0, self.capWords[capId]);
                # Register co-occurances
                for pair in it.product(self.capWords[capId], self.cliparts[sceneId]):
                    self.increaseCount(self.nWC0, pair);

        # Remove all the words that dont occur min number of times
        cutOff = [i for i in self.nW0 if self.nW0[i] < self.minOccurance];
        [self.nW0.pop(i, None) for i in cutOff];
        [self.nWC0.pop((i, c), None) for i in cutOff for c in self.nC0];

        cutOff = [i for i in self.nW1 if self.nW1[i] < self.minOccurance];
        [self.nW1.pop(i, None) for i in cutOff];
        [self.nWC1.pop((i, c), None) for i in cutOff for c in self.nC1];


    # Normalize counts to get probabilities
    def normalizeCounts(self):
        # Normalize 
        self.normalize(self.nW0);
        self.normalize(self.nW1);
        self.normalize(self.nC0);
        self.normalize(self.nC1);
        self.normalize(self.nWC0);
        self.normalize(self.nWC1);

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

        self.mi0 = defaultdict(float); 
        self.mi1 = defaultdict(float);
        # Compute mutual information and then find max
        for (w, c) in self.nWC0:
            self.mi0[(w, c)] = self.nWC0[(w, c)] * math.log(self.nWC0[(w, c)]/\
                                                (self.nW0[w] * self.nC0[c]));

        # Compute mutual information and then find max
        for (w, c) in self.nWC1:
            self.mi1[(w, c)] = self.nWC1[(w, c)] * math.log(self.nWC1[(w, c)]/\
                                                (self.nW1[w] * self.nC1[c]));

    # Get the alignment for all the train tuples at once, a wrapper for alignClipart
    # Input:
    #   tuples : dictionary of caption id : tuples
    def getTupleAlignment(self, tuples):
        # Scenes that have tuples
        nonZeroId = [i for i in tuples if len(tuples[i]) > 0];

        iterId = 0; # Premature stop
        for capId in nonZeroId:
            sceneId = capId/5;
            print self.cliparts[sceneId]
            print sceneId
            print self.capWords[capId];

            for tup in tuples[capId]:
                (clipP, clipS) = self.alignClipart(tup[0::2], \
                            self.cliparts[sceneId], self.types[sceneId]);
                print '%s : (%s, %s)' % (tup, clipP, clipS)
            print '\n'

            iterId += 1;
            if iterId > 20:
                break;
        #pdb.set_trace();

    # Compute the alignment for a (list of) word, given the word, sceneType and cliparts in the scene
    def alignClipart(self, word, cliparts, sceneType):
        if(isinstance(word, list)):
            bestAlign = tuple([self.alignClipart(w, cliparts,sceneType) \
                                                        for w in word]);
        else:
            if sceneType:
                mi = np.array(self.getCount(self.mi1, [(word, c) for c in cliparts]));
                bestAlign = cliparts[np.argmax(mi)]
            else:
                mi = np.array(self.getCount(self.mi0, [(word, c) for c in cliparts]));
                bestAlign = cliparts[np.argmax(mi)]
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

    # Compute mutual information between words and clipart
    #align.computeMI();

    # Saving the pickle for align
    savePath = dataPath + 'vqa_captions_mi.pickle';
    #with open(savePath, 'w') as dataFile:
    #    pickle.dump(align, dataFile);
    #print 'Saved at : %s' % savePath
        
    # Load the pickle for align
    with open(savePath, 'r') as dataFile:
        align = pickle.load(dataFile);
    print 'Loaded model from: %s' % savePath

    tuplesPath = dataPath + 'vqa_train_tuples.pickle';
    with open(tuplesPath, 'r') as fileId:
        tuples = pickle.load(fileId);
    # Get the alignments for the training tuples
    align.getTupleAlignment(tuples);

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
