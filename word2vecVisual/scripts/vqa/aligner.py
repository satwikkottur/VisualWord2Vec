# Script to read the cliparts, sentences and then align based on mutual 
# information
import cPickle as pickle
import numpy as np
import pdb
import itertools as it
import math

# Input:
#   1. List of training scenes (20000, currently)
#   2. Type of scene for each of the training scenes
#   3. List of cliparts present in the each of the training scenes
#   4. Captions for each scene
#   5. Image map between the captions and scenes
# 
# Output:
#   1. Alignment between the clipart and the words
#
# Test time:
#   1. Tuples from the caption sentences (to align)

class Aligner:
    # Setting up things
    minOccurance = 2;
    # Read the scene-caption map, scene types, scene clipart, 
    # tuples for the image
    def __init__(self):
        # Add data path
        dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';

        with open(dataPath + 'vqa_feature_map.txt', 'r') as fileId:
            self.maps = [i.strip('\n') for i in fileId.readlines()];
        with open(dataPath + 'scene_type.cPickle', 'r') as fileId:
            self.types = pickle.load(fileId);
        with open(dataPath + 'clipart_occurance.cPickle', 'r') as fileId:
            self.cliparts = pickle.load(fileId);
        with open(dataPath + 'vqa_train_tuples.pickle','r') as fileId:
            self.tuples = pickle.load(fileId);
        with open(dataPath + 'vqa_train_captions_lemma.txt', 'r') as fileId:
            self.capWords = [l.strip('\n').split(' ') for l in fileId.readlines()];
            
    # Collecting unique clipart objects and tuples, getting counts
    def computeCounts(self):
        # Count (co)-occurance for each caption and clipart (based on scene)
        # nWx : word occurance
        # nCx : clipart occurance
        # nWCx : word-clipart co-occurance
        self.nW0 = {}; self.nC0 = {}; self.nWC0 = {};
        self.nW1 = {}; self.nC1 = {}; self.nWC1 = {};

        # Weed out the words that dont occur more than some time
        for capId in xrange(0, len(self.capWords)):
            sceneId = int(self.maps[capId]);
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
        #pdb.set_trace();
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

        self.mi0 = {}; self.mi1 = {};
        self.align0 = {}; self.align1 = {};
        # Compute mutual information and then find max
        for (w, c) in self.nWC0:
            self.mi0[(w, c)] = self.nWC0[(w, c)] * math.log(self.nWC0[(w, c)]/\
                                                (self.nW0[w] * self.nC0[c]));

        # Compute mutual information and then find max
        for (w, c) in self.nWC1:
            self.mi1[(w, c)] = self.nWC1[(w, c)] * math.log(self.nWC1[(w, c)]/\
                                                (self.nW1[w] * self.nC1[c]));


    # Compute the alignment for all the scenes, a wrapper for alignClipart
    def computeAlignment(self):
        # Scenes that have tuples
        nonZeroId = [i for i in self.tuples if len(self.tuples[i]) > 0];

        iterId = 0;
        for capId in nonZeroId:
            sceneId = capId/5;
            print self.cliparts[sceneId]
            print sceneId
            print self.capWords[int(self.maps[capId])]

            for tup in self.tuples[capId]:
                clipP = self.alignClipart(tup[0], \
                            self.cliparts[sceneId], self.types[sceneId]);
                clipS = self.alignClipart(tup[2], \
                            self.cliparts[sceneId], self.types[sceneId]);

                print '%s : (%s, %s)' % (tup, clipP, clipS)

            print '\n'

            iterId += 1;
            if iterId > 20:
                break;
        #pdb.set_trace();
        #print self.mi0[(person, Doll)]

    # Compute the alignment for a word
    def alignClipart(self, word, cliparts, sceneType):
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
                if key in counter:
                    counter[key] += 1;
                else:
                    counter[key] = 1;
        else:
            key = keyList;
            if key in counter:
                counter[key] += 1;
            else:
                counter[key] = 1;

    # Handy function to check for a given list / element in counts
    # checks if a key exists, returns value if yes or 0 otherwise
    def getCount(self, counter, keyList):
        # Handle lists and single values separately
        if(isinstance(keyList, list)):
            counts = [];
            for key in keyList:
                if key in counter:
                    counts.append(counter[key]);
                else:
                    counts.append(0);
        else:
            key = keyList;
            if key in counter:
                counts = counter[key];
            else:
                counts = 0;

        return counts;
#**********************************************************************
        
if __name__ == '__main__':
    #align = Aligner();

    # Compute mutual information between words and clipart
    #align.computeMI();

    # For a given word and set of cliparts, get the alignment
    #align.computeAlignment();

    # Saving the pickle for align
    dataPath = '/Users/skottur/CMU/Personal/VisualWord2Vec/data/vqa/';
    #dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
    savePath = dataPath + 'vqa_captions_mi.pickle';
    #with open(savePath, 'w') as dataFile:
    #    pickle.dump(align, dataFile);
    #print 'Saved at : %s' % savePath
        
    # Load the pickle for align
    with open(savePath, 'r') as dataFile:
        align = pickle.load(dataFile);
    print 'Loaded model from: %s' % savePath

    # For a given word and set of cliparts, get the alignment
    align.computeAlignment();
    #align.alignClipart();

    #pdb.set_trace();
    # Print the alignment 
    '''for i in align.align0:
        print '%s : %s' % (i, align.align0[i])
    for i in align.align1:
        print '%s : %s' % (i, align.align1[i])'''

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
