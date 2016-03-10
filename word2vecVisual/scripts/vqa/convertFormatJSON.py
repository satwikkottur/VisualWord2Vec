# This script converts the scenes in the VQA format to ICCV format
# for localized feature extraction
import json
import pickle
import pdb
import sys

# Converts the scene JSON from VQA format into ICCV format
# for aligned feature extraction
#
# Input:
#   origPath : path to the orignal scene json file
#   alignment: the (P, R, S) alignment between cliparts
#   sceneClipart: extracted during alignment (cross-verification)
#   savePath : path to save the converted json file
def convertSceneJSON(origPath, alignment, sceneClipart, savePath):
    # Open the JSON file
    with open(origPath, 'r') as fileId:
        orig = json.load(fileId);
    
    # Make necessary changes
    new = orig['scene'];
    # Add few additional fields
    #      - primaryObject, primaryName, 
    #      - secondaryObject, secondaryName, 
    #      - relationName, init (not relevant ?)
    new['primaryName'] = alignment['tuple'][0];
    new['secondaryName'] = alignment['tuple'][2];
    new['relationName'] = alignment['tuple'][1];
    new['init'] = origPath.split('/')[-1];
    #new['primaryObject'] = {'idx':0, 'ins':0};
    #new['secondaryObject'] = {'idx':0, 'ins':0};

    # Get list of objects present
    present = [(j, new['availableObject'].index(i), i['instance'].index(j))\
                                        for i in new['availableObject'] \
                                        for j in i['instance'] \
                                        if j['present']];
    
    # Get the aligned cliparts for P, S
    pAlign = [alignment['P'] in i['name'] for (i, _, _) in present];
    sAlign = [alignment['S'] in i['name'] for (i, _, _) in present];

    if True not in pAlign or True not in sAlign:
        # Something is wrong!
        sys.exit(0);
    else:
        pClipart = present[pAlign.index(True)];
        new['primaryObject'] = {'idx': pClipart[1], 'ins': pClipart[2]};

        sClipart = present[sAlign.index(True)];
        new['secondaryObject'] = {'idx': sClipart[1], 'ins': sClipart[2]};

    # Save the converted file
    json.dump(new, open(savePath, 'w'));

###############################################################################
if __name__ == '__main__':
    dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
    alignPath = dataPath + 'vqa_train_alignment.pickle';

    # Read the alignment pickle
    with open(alignPath, 'r') as fileId:
        alignment = pickle.load(fileId);

    # Read a tuple with scene json and alignment
    scenePath = '/srv/share/vqa/release_data/abstract_v002/' + \
                                'scene_json/scene_indv/%d.json';
    savePath = dataPath + 'iccv_json/%d.json';

    # Also write a dictionary of tuples and corresponding features
    tupleDictPath = dataPath + 'vqa_tuples_train_dict.txt';
    dictId = open(tupleDictPath, 'w');

    # scenes with nonzero tuples
    tupleId = 0;
    nonZero = [i for i in alignment if len(alignment[i]) > 0];
    for sceneId in nonZero:
        print 'Converting scene : %d' % sceneId
        for caption in alignment[sceneId]['captions']:
            for tup in caption['tuples']:
                # Convert the scene json
                convertSceneJSON(scenePath % sceneId, tup, \
                        alignment[sceneId]['clipart'], savePath % tupleId);
                # Register the tuple in the dictionary
                dictId.write('<%s:%s:%s>(%d,%d)\n' % (tup['tuple'][0], \
                                                    tup['tuple'][1], \
                                                    tup['tuple'][2], \
                                                    tupleId, sceneId));
                tupleId += 1;


    dictId.close();
