# Script to lemmatize P,R,S tuples that were changed manually
import json
import re

# Read the map
jsonPath = '/home/satwik/VisualWord2Vec/data/spellings_relations.json';
with open(jsonPath) as dataFile:
    data = json.load(dataFile);

# Get the inverse map
inverseMap = {};
for i in data:
    inverseMap[data[i]] = i;

# Read the PRS tuple file
psrPath = '/home/satwik/VisualWord2Vec/data/PSR_features.txt';
psrFile = open(psrPath, 'rb');
psrLines = [re.split(':', i.strip('<>\n')) for i in psrFile.readlines()];

# Correct each tuple, if applicable
for tupl in psrLines:
    for j in tupl:
        if(j in inverseMap):
            psrLines[psrLines.index(tupl)][tupl.index(j)] = inverseMap[j] 

# Write to new file
savePath = '/home/satwik/VisualWord2Vec/data/PSR_features_lemma.txt';
saveFile = open(savePath, 'wb');

for tupl in psrLines:
    saveFile.write('<%s:%s:%s>\n' % tuple(tupl));

# close both the files
saveFile.close();
psrFile.close();

