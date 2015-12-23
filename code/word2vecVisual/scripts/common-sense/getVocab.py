# Script to get the vocabulary for P,R,S common sense task
import re

# Read the file
dataPath = '/home/satwik/VisualWord2Vec/data/PSR_features_lemma.txt'

with open(dataPath, 'rb') as dataFile:
    lines = [re.match('<([^:<>]*):([^:<>]*):([^<>:]*)>\n', i) \
                    for i in dataFile.readlines()];
    tokens = set([i.group(j).lower() for j in [1, 2, 3] for i in lines]);

# Write the tokens to file
fileId = open('/home/satwik/VisualWord2Vec/data/PSR_tokens.txt', 'wb');
[fileId.write('%s\n' % i) for i in tokens];
fileId.close();
