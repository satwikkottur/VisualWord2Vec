# Script to convert the data to format fit for visual word2vec
# author: Satwik Kottur
# email: skottur@andrew.cmu.edu

import os
import sys
import scipy.io as sio
import numpy as np
import pdb

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Error in usage:')
        print('python extractData.py <path to data> <path to store>(optional)');

    dataPath = sys.argv[1];
    if len(sys.argv) > 2:
        savePath = sys.argv[2];
    else:
        # Check for existance of folders and create accordingly
        savePath = '../../data/cs/';
        if not os.path.isdir('../../data/'): os.makedirs('../../data/');
        if not os.path.isdir(savePath): os.makedirs(savePath);

    print('Extracting data and saving at %s...' % savePath);

    # Reading the training data
    train = sio.loadmat(dataPath + 'data/PRS_abstract_scenes.mat');

    # Saving the PRS tuples
    print('Saving training PRS...');
    trainFile = open(savePath + 'PRS_train.txt', 'w');
    for tupleId in xrange(len(train['P'])):
        trainFile.write('<%s:%s:%s>\n' % \
                                (train['P'][tupleId][0][0],\
                                train['R'][tupleId][0][0],\
                                train['S'][tupleId][0][0]));
    trainFile.close();

    # saving the features
    print('Saving training visual features...');
    np.savetxt(savePath + 'visual_train.txt', train['fv'], comments='',\
        header='%d' % (train['fv'].shape[1],), delimiter=' ', fmt='%.6f');

    # Reading the test data
    test = sio.loadmat(dataPath + 'data/test.mat');

    # saving the test and validation features
    print('Saving testing PRS...');
    testFile = open(savePath + 'PRS_test.txt', 'w');
    for tupleId in xrange(len(test['P'])):
        groundTruth = 0;
        if test['score'][tupleId] > 0:
            groundTruth = 1;

        testFile.write('<%s:%s:%s> %d\n' % \
                            (test['P'][tupleId][0][0],\
                            test['R'][tupleId][0][0],\
                            test['S'][tupleId][0][0], groundTruth));
    testFile.close();

    # Reading the val data
    test = sio.loadmat(dataPath + 'data/val.mat');

    # saving the test and validation features
    print('Saving val PRS...');
    testFile = open(savePath + 'PRS_val.txt', 'w');
    for tupleId in xrange(len(test['P'])):
        groundTruth = 0;
        if test['score'][tupleId] > 0:
            groundTruth = 1;

        testFile.write('<%s:%s:%s> %d\n' % \
                            (test['P'][tupleId][0][0],\
                            test['R'][tupleId][0][0],\
                            test['S'][tupleId][0][0], groundTruth));
    testFile.close();

    # Reading the embeddings and storing in desirable format
    print('Saving word embeddings...');
    features = sio.loadmat(dataPath + 'data/coco_w2v.mat');

    tokens = [i[0] for i in features['tokens'][0]];
    data = np.column_stack((np.array(tokens), features['fv']));
    np.savetxt(savePath + 'word2vec_cs.bin', data, comments='',\
        header='%d %d' % features['fv'].shape, delimiter=' ', fmt='%s');
