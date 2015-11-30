# Script to compare and pick few examples for supplementary CVPR 2015
import cPickle as pickle
import numpy as np

# Print top query results, check who performs better
def printResult(before, after, tupleList, fileId):
    # Get top 5 results
    beforeTop = [];
    afterTop = [];
    for i in xrange(0, 5):
        beforeTop.append(before['score'].index(max(before['score'])));
        afterTop.append(after['score'].index(max(after['score'])));

        before['score'][beforeTop[i]] = -float('Inf');
        after['score'][afterTop[i]] = -float('Inf');

    # Print query, gt, top results, gtRank 
    fileId.write('Tuple:%s:%s\n' % (str(before['query']), str(before['ground'])))
    fileId.write('Before(%d):%s:%s:%s:%s:%s\n' % (before['gtRank'], \
                                            tupleList[beforeTop[0]], \
                                            tupleList[beforeTop[1]], \
                                            tupleList[beforeTop[2]], \
                                            tupleList[beforeTop[3]], \
                                            tupleList[beforeTop[4]]));
    fileId.write('After(%d):%s:%s:%s:%s:%s\n' % (after['gtRank'], \
                                            tupleList[afterTop[0]], \
                                            tupleList[afterTop[1]], \
                                            tupleList[afterTop[2]], \
                                            tupleList[afterTop[3]], \
                                            tupleList[afterTop[4]]));
     

beforeFile = 'before_scores_redone.pickle';
afterFile = 'after_scores.pickle';
tupleFile = 'tupleList.pickle';

with open(beforeFile, 'rb') as dataFile:
    before = pickle.load(dataFile);
with open(afterFile, 'rb') as dataFile:
    after = pickle.load(dataFile);
with open(tupleFile, 'rb') as dataFile:
    tupleList = pickle.load(dataFile);

# Randomly sample few queries
noChoices = 40;
sampled = list(np.random.choice(before.keys(), noChoices));

# Open two files to save the results separately
beforePath = 'results_before.txt';
afterPath = 'results_after.txt';

bId = open(beforePath, 'wb');
aId = open(afterPath, 'wb');

# See who performs better
for i in sampled:
    print 'Sampled :%d\n' % i
    # Get top queries
    if(before[i]['gtRank'] >= after[i]['gtRank']):
        printResult(before[i], after[i], tupleList, aId);
    else:
        printResult(before[i], after[i], tupleList, bId);

bId.close();
aId.close();
