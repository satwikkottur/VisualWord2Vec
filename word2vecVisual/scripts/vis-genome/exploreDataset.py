# Script to explore the datasets and splitting the features corresponding
import pdb
import os
import re
from collections import defaultdict

# Cross check the features with actual images
def crossCheckImages():
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';

    # Get the list of the images
    listing = os.listdir(dataPath + 'images/')

    # Read the features file
    featPath = dataPath + 'fc7_train_vectors.txt';
    with open(featPath, 'r') as fileId:
        line = [int(i.split('\t')[0]) for i in fileId.readlines()];

# Cross check the features with image list
def crossCheckImageLists():
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';

    # First read the image ids in the image list, features
    for ind in xrange(11):
        print 'Currently in %02d file...' % ind
        imgId = open(dataPath + 'imageListSplits/%02d' % ind, 'r');
        featId = open(dataPath + 'featListSplits/%02d' % ind, 'r');

        lineImg = [int(i.split('\t')[0]) for i in imgId.readlines()];
        lineFeat = [int(i.split('\t')[0]) for i in featId.readlines()];
    
        # Match and write the additional ones separately
        # the last 10 miss out in every file, store them separately
        missedInd = list(set(lineImg) - set(lineFeat));

        imgId.seek(0);
        imageList = [i.strip('\n').split('\t') for i in imgId.readlines()\
                                    if int(i.split('\t')[0]) in missedInd];
        imageDict = {int(i[0]):i[1] for i in imageList};

        with open(dataPath + 'featListSplits/extra_%02d' % ind, 'w') as saveId:
            [saveId.write('%d\t%s\n' % (i, imageDict[i])) for i in imageDict];

# Extracting the features for training on data
def prepareTrainDataset():
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';
    for ind in xrange(1):
        imgPath = dataPath + 'imageListSplits/%02d' % ind;
        featPath = dataPath + 'featListSplits/all_%02d' % ind;
        
        # Image labels
        with open(imgPath, 'r') as imgId:
            imgLabels = [int(i.split('\t')[0]) for i in imgId.readlines()];

        # Feature labels
        with open(featPath, 'r') as featId:
            featList = [i.strip('\n').split('\t') for i in featId.readlines()];
            feats = {int(i[0]):i[1] for i in featList};

        pdb.set_trace();

# Reading the image captions
def readTrainCaptions(captionPath):
    # Remove the new lines
    with open(captionPath, 'r') as fileId:
        captions = [i.strip('\n') for i in fileId.readlines()];

    newCaptions = [i for i in captions if len(i) > 0];

    with open(captionPath, 'w') as fileId:
        [fileId.write(i + '\n') for i in newCaptions];

    print 'Done removing the new lines'
    return

    with open(captionPath, 'r') as fileId:
        rawCaps = [i.strip('\n') for i in fileId.readlines()];
        capList = [re.search('([^: ]*) : ([^: ]*) : ([^:]*)', i) for i in rawCaps];

    pdb.set_trace();

    # Construct the dictionary
    captions = defaultdict(dict);
    for entry in capList:
        # Extract the image id and region id
        imgId = int(entry.group(1));
        regId = int(entry.group(2));

        captions[imgId][regId] = entry.group(3);

    pdb.set_trace();

###############################################################################
if __name__ == '__main__':
    # Call appropriate function (equivalent to multiple independent scripts)
    #crossCheckImageLists();

    #prepareTrainDataset();

    captionPath = '/home/satwik/VisualWord2Vec/data/vis-genome/genome_train_captions.txt';
    readTrainCaptions(captionPath);

