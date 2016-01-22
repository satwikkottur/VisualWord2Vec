# Script to explore the datasets and splitting the features corresponding
import pdb
import os
import re
import pickle
import sys
from collections import defaultdict
import numpy as np
import multiprocessing

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
def prepareTrainDataset(workerId):
    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';

    # Pre-load the dictionary for captions
    captionPath = dataPath + 'caption_dictionary.pickle';
    if os.path.exists(captionPath):
        with open(captionPath, 'r') as fileId:
            captions = pickle.load(fileid);
    else:
        # Re-compute again if doens't exist
        captionPath = dataPath + 'captionSplits/genome_train_captions.txt';
        captions = readTrainCaptions(captionPath);

    # Reading image index
    with open(dataPath + 'images/index.txt') as fileId:
        lines = [re.search('([^: ]*) : ([^: ]*)', i.strip('\n')) \
                                            for i in fileId.readlines()];
        imgIndDict = {int(i.group(1)):int(i.group(2)) for i in lines};

    # Read each chunk of features and images split
    imgPath = dataPath + 'imageListSplits/%02d' % workerId;
    featPath = dataPath + 'featListSplits/all_%02d' % workerId;
    capSavePath = dataPath + 'train/text_features_%02d' % workerId;
    featSavePath = dataPath + 'train/vis_features_%02d' % workerId;
    
    # Image labels
    with open(imgPath, 'r') as imgId:
        imgLabels = [int(i.split('\t')[0]) for i in imgId.readlines()];

    # Feature labels
    with open(featPath, 'r') as featId:
        featList = [i.strip('\n').split('\t') for i in featId.readlines()];
        feats = {int(i[0]):i[1] for i in featList};

    # For each image label, extract the corresponding image id, region id
    imgId = [i/10000 for i in imgLabels];
    imgId = [i/10000 * 1000 + i % 10000 for i in imgId];
    regId = [i % 10000 for i in imgLabels];

    ############## Sanity check #############
    # All image ids must be present
    # All region ids must be present
    absentImgId = [i for i in imgId if i not in imgIndDict];
    absentRegId = [i for i in xrange(len(regId)) \
                    if regId[i] not in captions[imgIndDict[imgId[i]]]];
    if len(absentImgId) or len(absentRegId):
        print 'Error in the image or region ids'
        sys.exit(0);
    #########################################
    
    capId = open(capSavePath, 'w');
    #featId = open(featSavePath, 'w');

    allFeatures = [];
    # Writing the captions and features
    for ii in xrange(len(imgId)):
        # Print after few iterations
        if ii % 100 == 0:
            print 'Current image(%d) : %d / %d' % (workerId, ii, len(imgId))
        # Caption
        capId.write(captions[imgIndDict[imgId[ii]]][regId[ii]] + '\n');
        # Feature
        feature = np.array([float(i) for i in \
                                    feats[imgLabels[ii]][1:-1].split(',')]);
        allFeatures.append(feature);

    # Close the caption file
    capId.close();
    #  Save using numpy
    np.savetxt(featSavePath, np.array(allFeatures), fmt='%0.6f', delimiter=' ',\
                    header=str(len(allFeatures[0])), comments='');

# Wrapper for the multi processign preparation of the dataset
def prepareTrainDatasetMulti(noThreads):
    jobs = [];
    for i in xrange(0, noThreads):
        thread = multiprocessing.Process(target=prepareTrainDataset, args=(i, ));
        jobs.append(thread);
        thread.start();

# Reading the image captions
def readTrainCaptions(captionPath):
    # Read the captions line
    with open(captionPath, 'r') as fileId:
        rawCaps = [i.strip('\n') for i in fileId.readlines()];
        capList = [re.search('([^: ]*) : ([^: ]*) : ([^:]*)', i) for i in rawCaps];

    # Construct the dictionary
    captions = defaultdict(dict);
    for entry in capList:
        # Extract the image id and region id
        imgId = int(entry.group(1));
        regId = int(entry.group(2));
        # Include the dictionary
        captions[imgId][regId] = entry.group(3);

    return captions;

# Change the visual feature format to suite svm format
def changeVisualFeatureFormat(inPath, outPath):
    inFile = open(inPath, "r");
    outFile = open(outPath, "w");

    # Read the lines and write in specific form
    with open(inPath, "r") as fileId:
        lines = [i.strip('\n').split(' ') for i in fileId.readlines()];

    # For each line, check if zero or not, and record
    ind = 0;
    for line in lines[1:]:
        outFile.write('1 ');
        ind += 1;
        # Print after every 5000 lines
        if ind % 5000 == 0:
            print '%d / %d' % (ind, len(lines))
        for i in xrange(len(lines[1]) - 1):
            if float(line[i - 1]) != 0.0:
                outFile.write(str(i+1) + ':' + line[i - 1] + ' ');

        # Always have the last feature
        outFile.write(str(len(lines[1])) + ':' + line[-1] + '\n')
    
    outFile.close();

def changeClusterFileFormat(inPath, outPath):
    # Open the in file and read the clsuters (ignore the original class)
    with open(inPath, 'r') as fileId:
        ids = [int(i.split('\t')[0].split(' ')[0]) for i in fileId.readlines()];

    with open(outPath, 'w') as fileId:
        [fileId.write(str(i) + '\n') for i in ids];

# This script adds appropriate headers to the visual features files, and saves them as is
#def addHeaders(inPath):
    # Read and add the 
    

###############################################################################
if __name__ == '__main__':
    # Call appropriate function (equivalent to multiple independent scripts)
    #crossCheckImageLists();

    # Preparing the data through multiple threads
    #prepareTrainDatasetMulti(11);

    dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/';
    #captionPath = dataPath + 'captionSplits/genome_train_captions.txt';
    #captions = readTrainCaptions(captionPath);

    ## Saving for easier access
    #with open(dataPath + 'caption_dictinary.pickle', 'w') as fileId:
    #    pickle.dump(captions, fileId);

    # Change the format of the visual features
    #for i in xrange(1, 11):
    #    print 'Current file : %d' % i
    #    inPath = dataPath + 'train/vis_features_%02d';
    #    outPath = dataPath + 'train/vis_features_%02d_libsvm';
    #    changeVisualFeatureFormat(inPath % i, outPath % i);

    # Change the format of the cluster file
    for i in xrange(11):
        print 'Current file : %d' % i
        inPath = dataPath + 'train/cluster_assignment_%02d_10k';
        outPath = dataPath + 'train/cluster_C_%02d_10k';
        changeClusterFileFormat(inPath % i, outPath % i);
