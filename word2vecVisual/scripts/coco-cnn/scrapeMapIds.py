# Script to strip off the maps
import re

def scrapeMapId(sourcePath, destPath):
    # Read the source lines, remove the leading number and colon
    # Print the remaining sentence
    with open(sourcePath, 'rb') as dataFile:
        lines = [i.strip('\n') for i in dataFile.readlines()];

    captions = [re.match('\d* : (.*)', i).group(1) for i in lines];

    # Debug check
    if [i for i in xrange(0, len(captions)) if \
                        len(lines[i]) - len(captions[i]) > 8] != []:
        print 'Some error'

    # Now dump the captions
    saveId = open(destPath, 'wb');
    [saveId.write(i + '\n') for i in captions];
    saveId.close();
##############################################################
dataPath = '/home/satwik/VisualWord2Vec/data/coco-cnn/';
sourcePath = dataPath + 'captions_coco_train_lemma.txt';
destPath = dataPath + 'captions_coco_train_lemma_nomaps.txt';

scrapeMapId(sourcePath, destPath);
