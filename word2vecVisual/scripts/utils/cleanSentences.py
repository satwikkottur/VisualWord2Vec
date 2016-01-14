# Script to remove the trailing and leading white space and newline characters

dataPath = '/home/satwik/VisualWord2Vec/data/vis-genome/train/';

for id in xrange(11):
    print 'Current file : %d' % id
    with open(dataPath + 'text_features_%02d' % id, 'r') as fileId:
        lines = [i.strip('\n').strip() for i in fileId.readlines()];

    with open(dataPath + 'text_features_%02d' % id, 'w') as fileId:
        [fileId.write(i + '\n') for i in lines];

