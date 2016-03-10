# Script to explore a dataset
import cPickle as pickle
import pdb

dataPath = '/home/satwik/VisualWord2Vec/data/vqa/';
#with open(dataPath + 'vqa_tuples_train_dict.txt', 'r') as fileId:
#    train = pickle.load(fileId);

# Get the tuples
#tuples = [tup for item in train.items() if len(item[1]) > 0 for tup in item[1]];

# read the alignment for training using vqa
with open(dataPath + 'vqa_train_alignment.pickle', 'r') as fileId:
    train = pickle.load(fileId);

# Get the non-empty tuples
tuples = [(tup, val['clipart']) for key, val in train.items() \
            if len(val['captions']) > 0 for tup in val['captions']];

fileId = open(dataPath + 'tuple_alignment.html', 'w');

# Header
header = '<html><head><style>table#t01{width:100%; background-color:#fff}';
header = header + 'table#t01 tr:nth-child(even){background-color:#ccc;}';
header = header + 'table#t01 tr:nth-child(odd){background-color:#fff;}';
header = header + 'table#t01 th{background-color:black;color:white}</style>';
header = header + '</head><body><h1>Tuple Alignment</h1><table id="t01">';
fileId.write(header);

# Write the table contents
for tupData in tuples:
    string = '<tr><td>%s</td><td>%s</td><td>%s</td><td>%s = %s</td><td>%s = %s</td></tr>\n';

    tup = tupData[0]['tuples'][0];
    string = string % (tupData[0]['caption'], tupData[1], tup['tuple'],\
                        tup['tuple'][0], tup['P'], \
                        tup['tuple'][2], tup['S']);
    fileId.write(string)

# Footer
footer = '</table></body></html>';
fileId.write(footer);

# close the file
fileId.close();
