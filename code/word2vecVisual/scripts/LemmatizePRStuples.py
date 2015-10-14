# Script to lemmatize P,R,S tuples that were changed manually
import json
import re
from nltk.stem import WordNetLemmatizer

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

lmt = WordNetLemmatizer();
# Correct each tuple, if applicable
for tupl in psrLines:
    # Find R in the inversemap but lemmatize P,S as nouns
    # P
    psrLines[prsLines.index(tupl)][0] = lmt.lemmatize(tupl[0]);

    # S
    psrLines[prsLines.index(tupl)][1] = lmt.lemmatize(tupl[1]);

    # R
    r = tupl[2];
    if r in inverseMap:
        psrLines[prsLines.index(tupl)][2] = inverseMap[r];

# Write to new file
savePath = '/home/satwik/VisualWord2Vec/data/PSR_features_lemma.txt';
saveFile = open(savePath, 'wb');

for tupl in psrLines:
    saveFile.write('<%s:%s:%s>\n' % tuple(tupl));

# close both the files
saveFile.close();
psrFile.close();

