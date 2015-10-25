# Script to read the tuples for the text based image retrieval task
import pickle
import os
from ImageRetriever import ImageRetriever

# Setting up paths
picklePath = '/home/satwik/VisualWord2Vec/data/text_retrieval_pilot_lemma.p';
tuplesData = pickle.load(open(picklePath, 'rb'));

# Order : Relation, Primary, Secondary
tuples = tuplesData['data'];

# Create instance of the task
task = ImageRetriever();

for i in tuples.keys()[0:5]:
    imgTag = i[0];
    task.retrieveImage(imgTag);

    # Get tuples
    print tuples[i][0]

# word2vec path
embedPath = '/home/satwik/VisualWord2Vec/models/vp_coco_embeddings.bin';
task.readWord2Vec(embedPath);
