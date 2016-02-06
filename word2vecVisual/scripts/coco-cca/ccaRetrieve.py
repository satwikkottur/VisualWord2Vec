# Script to read the CCA features and perform retrieval using multithreads

import numpy as np
import multiprocessing
import pdb

# Main thread 
# Input:
#   
def computeGroundTruthRank(testFeats, trainFeats, gTruths, workerId):
    # Return ranks
    rankList = [];
    # Use the similarity (dot product)
    for testId in xrange(len(testFeats)):
        if testId % 10 == 0:
            print 'Currently processing (%d) : %d / %d...' % \
                                        (workerId, testId, len(testFeats))

        # Compute the score with train feats and get the rank of ground truth
        scores = np.array([testFeats[testId].dot(train) for train in trainFeats]);

        temp = scores.argsort()[::-1]
        rankList.append(np.where(temp == gTruths[testId])[0][0] + 1);
        #ranks = np.empty(len(scores), int)
        #ranks[temp] = np.arange(len(scores))
        #ranks[gTruths[testId]]

        # Get the rank of the ground truth
     
    template = 'ranks/vp/ranks_%02d_before.txt';
    #template = 'ranks/mscoco/ranks_%02d_before.txt';
    # Write it to a file
    with open(template % workerId, 'w') as fileId:
        [fileId.write(str(i) + '\n') for i in rankList];

# Function to compute the ranks using multiprocessing
def getRanks():
    dataPath = '/home/satwik/VisualWord2Vec/data/vp/coco/';

    train = np.loadtxt(dataPath + 'train_caption_embeds_before.txt', skiprows=1);
    test = np.loadtxt(dataPath + 'test_caption_embeds_before.txt', skiprows=1);
    truth = np.loadtxt(dataPath + 'test_caption_maps.txt', int)

    # Normalize all the vectors (test and train)
    for i in xrange(len(train)):
        train[i] = train[i]/np.linalg.norm(train[i]);
    for i in xrange(len(test)):
        test[i] = test[i]/np.linalg.norm(test[i]);

    print 'Done reading test and train files...'

    # Setting up multiple workers
    noWorkers = 30;
    jobs = [];
    for i in xrange(noWorkers):
        p = multiprocessing.Process(target=computeGroundTruthRank, \
                        args=(test[i::noWorkers], train, truth[i::noWorkers], i));
        jobs.append(p)
        p.start()

# Function to compute the recall and other stats 
def getStats():
    saveFormat = 'ranks/vp/ranks_%02d_after.txt';
    #saveFormat = 'ranks/vis-genome/mscoco/ranks_%02d_before_00.txt';
    noFiles = 30;

    # Obtain all the ranks 
    ranks = [];
    for i in xrange(noFiles):
        with open(saveFormat % i, 'r') as fileId:
            ranks.extend([int(i.strip('\n')) for i in fileId.readlines()]);

    ranks = np.array(ranks);
    # Compute recall 1, recall 5, recall 10, median recall
    medRecall = np.median(ranks);

    recall1 = len(np.where(ranks <= 1)[0])/float(len(ranks));
    recall5 = len(np.where(ranks <= 5)[0])/float(len(ranks));
    recall10 = len(np.where(ranks <= 10)[0])/float(len(ranks));

    stats = (recall1, recall5, recall10, medRecall);

    # Print and return the statistics
    print stats
    return stats;
###############################################################
if __name__ == '__main__':
    #getRanks();

    getStats();

################################################################
# Additional information
#dataPath = '/home/satwik/VisualWord2Vec/data/coco-cca/mscoco/';
#
#train = np.loadtxt(dataPath + 'train_caption_embeds_after.txt', skiprows=1);
#test = np.loadtxt(dataPath + 'test_caption_embeds_after.txt', skiprows=1);
#truth = np.loadtxt(dataPath + 'test_caption_maps.txt', int)
