# Script to convert a json file into text file
# Takes a json file as input
# Spits out txt file

import json

# Input json path
jsonPath = '/home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_tokenized_stops.json';
content = json.load(open(jsonPath));

# Output text path
#txtPath = '/home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_tokenized.txt';
txtPath = jsonPath.replace('.json', '.txt');
txtFile = open(txtPath, 'wb');

# Iterating and writing each line in the text path
for i in content.keys():
    for j in content[i]:
        txtFile.write(j.encode('ascii', 'ignore') + '\n');
        #txtFile.write(str(j) + ' ');
   
txtFile.close();

