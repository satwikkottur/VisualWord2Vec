# A bunch of utilitiy functions for exploring and training on VQA dataset
import re
import cPickle as pickle
import numpy as np

def extractSceneClipart(featPath, savePath):
    listing = '%d_instances-random.cpickle';
    noTrain = 20000;

    clipart = {};
    sceneType = {};
    # Save the occurances for each of the clipart type
    for fileId in xrange(0, noTrain):
        with open(featPath + listing % fileId, 'rb') as dataFile:
            feature = pickle.load(dataFile);

        # Get clipart type from only the presence features
        #clipart[fileId] = [re.search('([^\d.]*)\d*.\d', i['name'][9:]).group(1) \
        #                                    for i in feature \
        #                                    if ('presence' in i['tags'])\
        #                                        and (i['feature'][0] == 1)];
        sceneType[fileId] = feature[0]['feature'][0];

        print 'Extracting clipart : %d / %d' % (fileId, noTrain);
    # Save the cliparts
    with open(savePath, 'wb') as dataFile:
        pickle.dump(sceneType, dataFile);

##############################################################################
##############################################################################
featPath = '/srv/share/vqa/release_data/abstract_v002/scene_json/features_v2/metafeatures/';
savePath = '/home/satwik/VisualWord2Vec/data/vqa/scene-type.cPickle';
extractSceneClipart(featPath, savePath);
'''
loadPath = savePath;

with open(loadPath, 'rb') as dataFile:
    clipart = pickle.load(dataFile);

for fileId in xrange(0, 10):
    with open(featPath + '%d_instances-random.cpickle' % fileId, 'rb') as dataFile:
        feat = pickle.load(dataFile);
    print feat[0]['feature'][0]'''
