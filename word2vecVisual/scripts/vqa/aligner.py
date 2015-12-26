# Script to read the cliparts, sentences and then align based on mutual 
# information
import cPickle as pickle
import numpy as np
import pdb
import itertools as it

# Input:
#   1. List of training scenes (20000, currently)
#   2. Type of scene for each of the training scenes
#   3. List of cliparts present in the each of the training scenes
#   4. Tuples for each of the caption sentences
#   5. Image map between the captions and scenes
# 
# Output:
#   1. Alignment between the clipart and the words

class Aligner:
    # Setting up things
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

    # Collecting unique clipart objects and tuples, getting counts
    def computeCounts(self):
        # Unique clipart (scene0, scene1) and words (P, S) from tuples
        self.art1 = [];
        self.art0 = [];
        for i in self.cliparts:
            if self.types[i]:
                self.art1.extend(self.cliparts[i]);
            else:
                self.art0.extend(self.cliparts[i]);
        self.art1 = set(self.art1);
        self.art0 = set(self.art0);

        self.word = [k for i in self.tuples.values() for j in i \
                                                for k in (j[0], j[2])];
        self.word = set(self.word);

        # Count (co)-occurance for each tuple and tuple (based on scene)
        # nWx : word occurance
        # nCx : clipart occurance
        # nWCx : Word-clipart co-occurance
        nW0 = {}; nC0 = {}; nWC0 = {};
        nW1 = {}; nC1 = {}; nWC1 = {};

        for capId in self.tuples:
            sceneId = int(self.maps[capId]);
            if self.types[sceneId]:
                # Register the clipart
                for i in self.cliparts[sceneId]:
                    self.increaseCount(nC1, i)

                # Register the word
                for i in self.tuples[capId]:
                    # Only primary and secondary
                    self.increaseCount(nW1, i[0])
                    self.increaseCount(nW1, i[2])

                    # Register co-occurances
                    for (w, c) in it.product(i[0::2], self.cliparts[sceneId]):
                        self.increaseCount(nWC1, (w, c));

            else:
                # Register the clipart
                for i in self.cliparts[sceneId]:
                    self.increaseCount(nC0, i)

                # Register the word
                for i in self.tuples[capId]:
                    self.increaseCount(nW0, i[0])
                    self.increaseCount(nW0, i[2])

                    # Register co-occurances
                    for (w, c) in it.product(i[0::2], self.cliparts[sceneId]):
                        self.increaseCount(nWC0, (w, c));

        #pdb.set_trace();

    # Compute the alignment between 

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
        
if __name__ == '__main__':
    align = Aligner();
    align.computeCounts();
    # Compute alignment between words and clipart
    align.computeAlignment();

