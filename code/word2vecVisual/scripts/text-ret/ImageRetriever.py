# Class to retrieve the image using text only
import os
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from copy import deepcopy
import sys

class ImageRetriever:
    # Attributes
    # Word2vec embeddings
    embeds = {};
    Rembeds = {};
    Sembeds = {};
    Pembeds = {};
    embedDim = 0;
    vocabSize = 0;
    tagTupleMap = {}; # Compute the ground tuple given the tag
    tupleIdMap = {}; # Associate each unique ground tuple with an id
    tupleList = []; # List of tuples to ensure some ordering
    single = True;
 
    # Constructor
    # Determines the mode of embeddings (single / multiple)
    def __init__(self, mode = 'SINGLE'):
        if mode == 'SINGLE':
            self.single = True;
        else:
            self.single = False;

    # Method to retrive the image
    def retrieveImage(self, imgTag):
        # Image collection path
        imgPath = '/srv/share/al/data_model/image_data/';

        imgName = imgPath + imgTag + '.png';
        # Check if existance of file
        if not os.path.isfile(imgName):
            print 'File not found!'

    # Read the word2vec embeddings (private)
    def __readWord2Vec(self, embedPath):
        embeds = {};

        # Reading the file
        with open(embedPath, 'rb') as dataFile:
            lines = [i.strip('\n') for i in dataFile.readlines()];

        # Read number of vocabsize and dimension
        regObj = re.match('(\d*) (\d*)', lines[0]);
        embedDim = int(regObj.group(2));
        vocabSize = int(regObj.group(1));

        for i in lines[1:]:
            iSplit = i.split(' ', 1)
            word = iSplit[0];
            embeds[word] = np.fromstring(iSplit[1], dtype = float, sep = ' ');

        return (embeds, embedDim, vocabSize);
        
    # Read word2vec
    def loadWord2Vec(self, embedPaths):
        if(self.single):
            # Read the word2vec for one embedding
            if isinstance(embedPaths, list):
                print 'Expected one path for single embedding!'
                sys.exit(0);

            (self.embeds, self.embedDim, self.vocabSize) = \
                                        self.__readWord2Vec(embedPaths);

        else:
            # Reading the file, each for p, r, s
            if(len(embedPaths) != 3):
                print 'Expected three paths for multi embeddings'
                sys.exit(0);

            # P
            (self.Pembeds, self.embedDim, self.vocabSize) = \
                                self.__readWord2Vec(embedPaths['p']);
            # R
            (self.Rembeds, embedDim, vocabSize) = \
                                self.__readWord2Vec(embedPaths['r']);
            if(embedDim != self.embedDim or vocabSize != self.vocabSize):
                # Raise alarm
                print 'Mismatch in dimensions or vocab size'
                sys.exit(0);

            # S
            (self.Sembeds, embedDim, vocabSize) = \
                                self.__readWord2Vec(embedPaths['s']);
            if(embedDim != self.embedDim or vocabSize != self.vocabSize):
                # Raise alarm
                print 'Mismatch in dimensions or vocab size'
                sys.exit(0);

        print 'Done reading the embeddings!'

        if len(self.tupleList) > 0:
            self.__getGroundEmbeddings()
        
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
            self.tagTupleMap[i.group(1)] = self.__lemmatizeTuples(tup);

        # Construct index map for ground tuples
        self.tupleList = list(set(self.tagTupleMap.values()));
        count = 0;
        for i in self.tupleList:
            self.tupleIdMap[i] = count;
            count = count + 1;

        # Check if embeddings are read, if yes compute ground tuples
        if(self.embedDim != 0):
            self.__getGroundEmbeddings();

        print 'Done reading the ground tuples!'

    # Compute embeddings for a given phrase
    def __computeEmbedding(self, phrase, embeds):
        # Gather the vector embeddings
        embedWord = np.array([embeds[i.lower()] for i in phrase.split(' ') if i.lower() in embeds]);

        # None of the words are in dict
        if len(embedWord) == 0:
            embedding = np.zeros(self.embedDim);
        else:
            embedding = np.mean(embedWord, axis = 0);
            # Normalize and return
            embedding = embedding/np.linalg.norm(embedding);

        return embedding;

    # Computing the embedding for a tuple (either single/multiple)
    def __computeTupleEmbedding(self, tup):
        if self.single:
            # Single embed
            embed = [self.__computeEmbedding(i, self.embeds) for i in tup];

        else:
            # Multi embeds
            embed = [];
            embed.append(self.__computeEmbedding(tup[0], self.Rembeds));
            embed.append(self.__computeEmbedding(tup[1], self.Pembeds));
            embed.append(self.__computeEmbedding(tup[2], self.Sembeds));

        return embed

    # Compute ground truth embeddings, avoid re-doing calculations
    def __getGroundEmbeddings(self):
        self.gtEmbed = {};

        # Compute for each ground tuple
        for tup in self.tupleIdMap:
            self.gtEmbed[tup] = self.__computeTupleEmbedding(tup);

    # Lemmatizing tuples
    def __lemmatizeTuples(self, tuples):
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
    def __scoreQueryTuple(self, qTuple, qTag):
        qTupEmbeds = self.__computeTupleEmbedding(qTuple);

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
        print self.tupleList[score.index(sortedScore[0])]
        print self.tupleList[score.index(sortedScore[1])]
        print self.tupleList[score.index(sortedScore[2])]
        print self.tupleList[score.index(sortedScore[3])]
        print self.tupleList[score.index(sortedScore[4])]'''
        #print gtRank, gtInd, gtScore
        #return self.gt[self.gt.keys()[predInd]]
        return gtRank

    # Perform the task
    def performTask(self, dataTuples):
        if self.single:
            print 'Starting task in %s mode' % 'SINGLE'
        else:
            print 'Starting task in %s mode' % 'MULTI'
        
        # Required Top@ recalls
        recInds = [1, 5, 10, 50, 100];
        recalls = dict.fromkeys(recInds, 0);
        ranks = []; # Get all the ranks to compute the median

        noTuples = len(dataTuples);
        # For each tuple, we have a corresponding gt
        count = 0; iterCount = 0;
        for i in dataTuples.keys():
            # indication of progress
            print 'Tuples : %d / %d' % (iterCount, noTuples)
            iterCount += 1;

            # Consider only top three
            for j in dataTuples[i][0:2]:
                gtRank = self.__scoreQueryTuple(j, i[0]);
                ranks.append(gtRank);
                count += 1;
                
                # Iteratively count towards recall
                for recId in recInds:
                    if gtRank < recId:
                        recalls[recId] += 1;
            
        # Print the results
        print '*********************'
        for i in recInds:
            print 'Recall (%d) : %f)' % (i, recalls[i]/float(count))
        print 'Med r: %d' % np.median(np.array(ranks))
        print '*********************'

    # Compute the gtRank for a test caption 
    def __scoreQueryCaption(self, qCap, qImgId):
        score = [];
        for i in self.capList:
            score.append(self.__computeEmbedding(qCap, self.embeds).dot(\
                                            self.groundEmbeds[i].transpose()));

        gtScore = score[self.capList.index(qImgId)];

        # Sort and find out the rank of gtScore
        score.sort(reverse=True);
        gtRank = score.index(gtScore);

        return gtRank;

    # Separate the train and test sets for easier evaluation
    def setupTrainTestCOCO(self, captions):
        # Pull out the first caption to serve as the ground truth
        self.groundCaps = {};
        self.groundEmbeds = {};
        self.capList = [];
        self.testCaps = {};

        for i in captions['annotations']:
            imgId = i['image_id'];
            # Capture the ground truth
            if(imgId not in self.groundCaps):
                self.capList.append(imgId);
                self.groundCaps[imgId] = i['caption'];
                self.testCaps[imgId] = [];

            # Capture the test tuples
            else:
                self.testCaps[imgId].append(i['caption']);

        '''print len(captions['annotations'])
        print len(set([i['image_id'] for i in captions['annotations']]))
        print len(self.groundCaps)
        print len(self.testCaps)
        print len(self.capList)'''

        # Compute the embeddings for the ground tuples
        self.groundEmbeds = {i: self.__computeEmbedding(self.groundCaps[i], self.embeds)\
                                for i in self.groundCaps};

    # Setup the system for multiprocessing
    def setupMultiProcessing(self, noThreads):
        self.recalls = [{} for i in xrange(0, noThreads)];
        self.ranks = [[] for i in xrange(0, noThreads)];

    # Perform the task in multiprocesesing way
    def performTaskMultiCOCO(self, threadId, noThreads):
        # Required Top@ recalls
        recInds = [1, 5, 10, 50, 100];
        recalls = dict.fromkeys(recInds, 0);
        ranks = []; # Get all the ranks to compute the median
        count = 0;
        iterCount = 0;

        testList = self.capList[threadId::noThreads];
        testCaps = {i:self.testCaps[i] for i in testList};

        # Compute similarity for each test case and get the ground truth rank
        for i in testCaps:
            # Print progress:
            print 'Caption (%d) : %d / %d' % (threadId, iterCount, len(testCaps))
            iterCount += 1;
                
            # For each query in the image
            for j in testCaps[i]:
                gtRank = self.__scoreQueryCaption(j, i);
                
                ranks.append(gtRank);
                count += 1;
                
                # Iteratively count towards recall
                for recId in recInds:
                    if gtRank < recId:
                        recalls[recId] += 1;
            
        # Print the results
        '''.print '*********************'
        for i in recInds:
            print 'Recall (%d) : %f)' % (i, recalls[i]/float(count))
        print 'Med r: %d' % np.median(np.array(ranks))
        print '*********************'''

        # Store the results
        self.recalls[threadId] = recalls;
        self.ranks[threadId] = ranks;

    # Perform the task for COCO
    def performTaskCOCO(self, captions):
        self.setupTrainTestCOCO(captions);

        # Required Top@ recalls
        recInds = [1, 5, 10, 50, 100];
        recalls = dict.fromkeys(recInds, 0);
        ranks = []; # Get all the ranks to compute the median
        count = 0;
        iterCount = 0;

        # Compute similarity for each test case and get the ground truth rank
        for i in self.testCaps:
            # Print progress:
            print 'Caption: %d / %d' % (iterCount, len(self.testCaps))
            iterCount += 1;
                
            # For each query in the image
            for j in self.testCaps[i]:
                gtRank = self.__scoreQueryCaption(j, i);
                
                ranks.append(gtRank);
                count += 1;
                
                # Iteratively count towards recall
                for recId in recInds:
                    if gtRank < recId:
                        recalls[recId] += 1;
            
        # Print the results
        print '*********************'
        for i in recInds:
            print 'Recall (%d) : %f)' % (i, recalls[i]/float(count))
        print 'Med r: %d' % np.median(np.array(ranks))
        print '*********************'





