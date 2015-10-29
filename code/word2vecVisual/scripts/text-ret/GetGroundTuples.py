# Script to get the ground truth tuples
import pickle
import os

# Setting up paths
picklePath = '/home/satwik/VisualWord2Vec/data/text_retrieval_pilot_lemma.p';
tuplesData = pickle.load(open(picklePath, 'rb'));

# Order : Relation, Primary, Secondary
tuples = tuplesData['data'];

for i in tuples.keys()[0:5]:
    imgTag = i[0];

    print imgTag

