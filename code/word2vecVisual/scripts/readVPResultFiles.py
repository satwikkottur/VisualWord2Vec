# This is a script to read the output files from the vp task
import re
    
def printResults(dumpPath):
    #dumpPath = '../modelsNdata/backup/out_val_sents';
    fileId = open(dumpPath, 'rb');
    if(fileId == None):
        print 'File not found!'
        return

    lines = [i.strip('\n') for i in fileId.readlines()];

    regExp = 'mAP: Test \(([\d.]*)\) Val\(([\d.]*)\)';
    splitLines = [re.search(regExp, i) for i in lines];

    valScores = [i.group(2) for i in splitLines if i is not None];
    testScores = [i.group(1) for i in splitLines if i is not None];

    print dumpPath
    print 'Baseline: %s\nBest VP: %s\n' % \
                (testScores[0], testScores[valScores.index(max(valScores))])
    fileId.close();

###########################################################################
#dumpPath = '../modelsNdata/backup/out_val_sents';
dumpPaths = ['../out_val1k_sents', \
            '../out_val1k_sents_pca', \
            '../out_val1k_words', \
            '../out_val1k_words_pca', \
            '../out_val1k_words_wiki', \
            '../out_val1k_sents_wiki'];

dumpPaths.append(['../out_enum_test_sents', \
            '../out_enum_test_desc', \
            '../out_enum_test_words']);

for i in dumpPaths:
    printResults(i);
