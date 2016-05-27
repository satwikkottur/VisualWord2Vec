# Script to read the tuples for the text based image retrieval task
import pickle
import json
from ImageRetriever import ImageRetriever
import sys

if __name__ == '__main__':
    if len(sys.argv) != 3 and len(sys.argv) != 5:
        print('Wrong usage:')
        print('SINGLE: python performRetrieval.py <path to data> <path to embedding file>');
        print('MULTI: python performRetrieval.py <path to data>' + \
                    ' <path P embed> <path R embed> <path S embed>');
        sys.exit(1);

    # Read the inputs
    if len(sys.argv) == 3:
        dataPath = sys.argv[1];
        embedPath = sys.argv[2];
        mode = 'SINGLE';

    elif len(sys.argv) == 5:
        dataPath = sys.argv[1];
        embedPath = [sys.argv[i] for i in [2, 3, 4]];
        mode = 'MULTI';

    print 'Reading the tuples...'
    # Setting up paths
    #dataPath = '/home/satwik/VisualWord2Vec/data/text-ret/';
    tupPath = dataPath + 'text_ret_tuples.pickle';
    with open(tupPath, 'r') as dataFile:
        tupData = pickle.load(dataFile);

    # Order : Relation, Primary, Secondary
    tuples = tupData['data'];

    # Create instance of the task (with multiple / single embeddings)
    task = ImageRetriever(mode);
    #task = ImageRetriever('SINGLE', raw = True);
    # Read the ground truth
    gtPath = dataPath + 'text_ret_gt.txt';
    task.readGroundTuples(gtPath);

    # Reading embeddings
    task.loadWord2Vec(embedPath)
    task.performTask(tuples)
