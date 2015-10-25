# Class to retrieve the image using text only
import os
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from copy import deepcopy

class ImageRetriever:
    # Attributes
    # Word2vec embeddings
    embeds = {};
    embedDim = 0;
    vocabSize = 0;
    tagTupleMap = {}; # Compute the ground tuple given the tag
    tupleIdMap = {}; # Associate each unique ground tuple with an id
    tupleList = []; # List of tuples to ensure some ordering
 
    # Constructor
    def __init__(self):
        pass
        # Word2vec embeddings
        #self.embeds = {};
        #self.embedDim

    # Method to retrive the image
    def retrieveImage(self, imgTag):
        # Image collection path
        imgPath = '/srv/share/al/data_model/image_data/';

        imgName = imgPath + imgTag + '.png';
        # Check if existance of file
        if not os.path.isfile(imgName):
            print 'File not found!'

    # Read the word2vec embeddings
    def readWord2Vec(self, embedPath):
        # Reading the file
        with open(embedPath, 'rb') as dataFile:
            lines = [i.strip('\n') for i in dataFile.readlines()];

        # Read number of vocabsize and dimension
        regObj = re.match('(\d*) (\d*)', lines[0]);
        self.embedDim = int(regObj.group(2));
        self.vocabSize = int(regObj.group(1));

        for i in lines[1:]:
            iSplit = i.split(' ', 1)
            word = iSplit[0];
            self.embeds[word] = np.fromstring(iSplit[1], dtype = float, sep = ' ');

        # Check if ground tuples exist, if yes compute ground tuples
        if len(self.tagTupleMap) != 0:
            self.getGroundEmbeddings();

        print 'Done reading the embeddings!'

    #Reading the ground truth tuples
    def readGroundTuples(self, gtPath):
        # Read the file
        with open(gtPath, 'rb') as dataFile:
            lines = [i.strip('\n') for i in dataFile.readlines()];

        regExp = '(\w*):<([^<>:]*):([^<>:]*):([^<>:]*)>';
        regObj = [re.match(regExp, i) for i in lines];

        for i in regObj:
            tup = (i.group(2), i.group(3), i.group(4));
            # Lemmatize tuple
            self.tagTupleMap[i.group(1)] = self.lemmatizeTuples(tup);

        # Construct index map for ground tuples
        self.tupleList = list(set(self.tagTupleMap.values()));
        count = 0;
        for i in self.tupleList:
            self.tupleIdMap[i] = count;
            count = count + 1;

        # Check if embeddings are read, if yes compute ground tuples
        if(self.embedDim != 0):
            self.getGroundEmbeddings();

        print 'Done reading the ground tuples!'

    # Compute embeddings for a given phrase
    def computeEmbedding(self, phrase):
        # Gather the vector embeddings
        embedWord = np.array([self.embeds[i.lower()] for i in phrase.split(' ') if i.lower() in self.embeds]);

        # None of the words are in dict
        if len(embedWord) == 0:
            embedding = np.zeros(self.embedDim);
        else:
            embedding = np.mean(embedWord, axis = 0);
            # Normalize and return
            embedding = embedding/np.linalg.norm(embedding);

        return embedding;

    # Compute ground truth embeddings, avoid re-doing calculations
    def getGroundEmbeddings(self):
        self.gtEmbed = {};

        for tup in self.tupleIdMap:
            self.gtEmbed[tup] = [self.computeEmbedding(i.lower()) for i in tup];

    # Lemmatizing tuples
    def lemmatizeTuples(self, tuples):
        lmt = WordNetLemmatizer();

        # Multiple tuples
        if isinstance(tuples, list):
            lemmaTuples = [(' '.join([lmt.lemmatize(c.lower(), 'v') for c in j[0].split(' ')]), \
                            ' '.join([lmt.lemmatize(c.lower(), 'n') for c in j[1].split(' ')]), \
                            ' '.join([lmt.lemmatize(c.lower(), 'n') for c in j[2].split(' ')])) \
                            for j in tuples];
        else:
            lemmaTuples = (' '.join([lmt.lemmatize(c.lower(), 'v') for c in tuples[0].split(' ')]), \
                            ' '.join([lmt.lemmatize(c.lower(), 'n') for c in tuples[1].split(' ')]), \
                            ' '.join([lmt.lemmatize(c.lower(), 'n') for c in tuples[2].split(' ')]));
        return lemmaTuples

    # Compute score wrt ground tuples, given a tuple
    def scoreQueryTuple(self, qTuple, qTag):
        # Compute individual scores (R, P, S)
        qTupEmbeds = [self.computeEmbedding(i.lower()) for i in qTuple];
        #print qTupEmbeds

        #score = np.zeros(len(self.gt));
        score = [];
        for i in xrange(0, len(self.tupleList)):
            gTupEmbeds = self.gtEmbed[self.tupleList[i]];

            score.append(np.sum([qTupEmbeds[j].dot(gTupEmbeds[j].transpose()) \
                                    for j in [0, 1, 2]]));

        # Get the ground truth tuple
        gtTuple = self.tagTupleMap[qTag];
        gtInd = self.tupleIdMap[gtTuple];
        gtScore = score[gtInd];

        # Find where gt is ranked
        sortedScore = deepcopy(score);
        sortedScore.sort(reverse=True);
        gtRank = sortedScore.index(gtScore);

        '''print '\nQuery : %s\nGround: %s' % (qTuple, gtTuple)
        print gtScore, sortedScore[0]
        print 'Ground rank: %d\nTop queries:' % gtRank
        print self.tupleIdMap.keys()[score.index(sortedScore[0])]
        print self.tupleIdMap.keys()[score.index(sortedScore[1])]
        print self.tupleIdMap.keys()[score.index(sortedScore[2])]
        print self.tupleIdMap.keys()[score.index(sortedScore[3])]
        print self.tupleIdMap.keys()[score.index(sortedScore[4])]'''
        #print gtRank, gtInd, gtScore
        #return self.gt[self.gt.keys()[predInd]]
        return gtRank

    # Perform the task
    def performTask(self, dataTuples):
        # Top 1, Top 5, Top 10 recalls
        recall1 = 0;
        recall5 = 0;
        recall10 = 0;
        noTuples = len(dataTuples);

        # For each tuple, we have a corresponding gt
        count = 0; iterCount = 0;
        for i in dataTuples.keys():
            # indication of progress
            print 'Tuples : %d / %d' % (iterCount, noTuples)
            iterCount += 1;

            for j in dataTuples[i]:
                gtRank = self.scoreQueryTuple(j, i[0]);
                count += 1;
                if gtRank == 0:
                    recall1 += 1;
                    recall5 += 1;
                    recall10 += 1;
                elif gtRank < 5:
                    recall5 += 1;
                    recall10 += 1;
                elif gtRank < 10:
                    recall10 += 1;

        print recall1/float(count), recall5/float(count), recall10/float(count)
