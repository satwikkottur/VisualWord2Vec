# A bunch of utilitiy functions for exploring and training on VQA dataset
import re
import cPickle as pickle
import numpy as np
import pdb

# Get the cliparts for the scenes
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
        # Get the scene type
        sceneType[fileId] = feature[0]['feature'][0];

        print 'Extracting clipart : %d / %d' % (fileId, noTrain);
    # Save the cliparts
    with open(savePath, 'wb') as dataFile:
        pickle.dump(sceneType, dataFile);

# Get the aligned features for the tuples using alignment
# Input:
#   Clipart features, alignment between tuples, path to save the tuples and features
# Output:
#   Two files - tuples and features
def extractAlignedFeatures(featPath, alignPath, savePath):
    listing = '%d_instances-random.cpickle';

    # Load the alignment between tuples and cliparts
    with open(alignPath, 'r') as fileId:
        align = pickle.load(fileId);

    # Scenes with tuples
    nonZero = [i for i in align if len(align[i]['captions']) > 0];
    for sceneId in nonZero[0:5]:
        # Read the entire features
        with open(featPath + listing % sceneId, 'r') as fileId:
            feature = pickle.load(fileId);

        # Get name of cliparts that exist
        clips = [i['name'][9:] for i in feature if ('presence' in i['tags'] and \
                                                    i['feature'][0])];

        # Get features corresponding to each of these 
        clipFeats = {i:[j for j in feature if i in j['tags']] for i in clips};

        pdb.set_trace();
        # Get the features for each of the aligned tuples
        for tup in align[sceneId]['captions']:
            print tup
            
##############################################################################
##############################################################################
if __name__ == '__main__':
    featPath = '/srv/share/vqa/release_data/abstract_v002/' + \
                        'scene_json/features_v2/metafeatures/';
    savePath = '/home/satwik/VisualWord2Vec/data/vqa/scene-type.cPickle';

    # Get the cliparts for the scenes
    #extractSceneClipart(featPath, savePath);

    # Get the aligned features for P, S
    dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
    alignPath = dataPath + 'vqa_train_alignment.pickle';
    extractAlignedFeatures(featPath, alignPath, dataPath);

    '''
    loadPath = savePath;

    with open(loadPath, 'rb') as dataFile:
        clipart = pickle.load(dataFile);

    for fileId in xrange(0, 10):
        with open(featPath + '%d_instances-random.cpickle' % fileId, 'rb') as dataFile:
            feat = pickle.load(dataFile);
        print feat[0]['feature'][0]'''
