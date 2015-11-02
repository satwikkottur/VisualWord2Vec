# This script reads the results from the multiprocessing setting of MS COCO 
# retrieval task
import re
import sys
import numpy as np

def readResults(resultPath):
    with open(resultPath, 'rb') as dataFile:
        lines = [i.strip('\n') for i in dataFile.readlines()];

    # Separate recall lines and med R lines
    regLines = [re.match('Med r \((\d*)\): \[([\d, ]*)\]', i) for i in lines];
    medianLines = [i for i in regLines if i is not None];

    regLines = [re.match('Recall \((\d*)\) \((\d*)\) : (\d*) / (\d*)', i) for i in lines];
    recallLines = [i for i in regLines if i is not None];

    ranks = [];
    # Append all the ranks
    for i in medianLines:
        threadRanks = [int(j) for j in i.group(2).split(', ')];
        ranks.extend(threadRanks);

    # Create a dictionary for counts, recalls in each thread
    counts = {};
    recalls = {};

    for i in recallLines:
        #print i.group(1), i.group(2), i.group(3), i.group(4)
        if i.group(1) in counts:
            if counts[i.group(1)] != int(i.group(4)):
                print 'Count mis-match'
                sys.exit(1);
        else:
            # Add the thread
            counts[i.group(1)] = int(i.group(4));
            recalls[i.group(1)] = {};
            
        recalls[i.group(1)][i.group(2)] = int(i.group(3));

    # Tally the counts in ranks and recalls
    totalCount = sum(counts.values());
    if(totalCount != len(ranks)):
        print 'Count mis-match'
        sys.exit(1);

    # Now compute the final recalls and ranks
    medianR = np.median(np.array(ranks));

    recallVals = dict.fromkeys(recalls.values()[0].keys(), 0);
    for i in recalls:
        for j in recalls[i]:
            recallVals[j] += recalls[i][j];

    # Final computation
    print '*********************'
    recallIds = ['1', '5', '10', '50', '100'];
    for i in recallIds:
        print 'Recall (%s) : %f' % (i, recallVals[i]/float(totalCount))
    print 'Med r : %d' % medianR
    print '*********************'''
######## END OF METHODS ################################

resultPath = 'out_coco_multi_baseline';
readResults(resultPath);
resultPath = 'out_coco_multi_refine';
readResults(resultPath);
resultPath = 'out_coco_multi_wiki_baseline';
readResults(resultPath);
resultPath = 'out_coco_multi_wiki_refined';
readResults(resultPath);
