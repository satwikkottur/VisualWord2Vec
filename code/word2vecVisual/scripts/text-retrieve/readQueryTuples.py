# Script to read the tuples for the text based image retrieval task
import pickle
import os
from ImageRetriever import ImageRetriever

# Setting up paths
dataPath = '/home/satwik/VisualWord2Vec/data/text-retrieval/';
picklePath = dataPath + 'text_retrieval_pilot_lemma.p';
tuplesData = pickle.load(open(picklePath, 'rb'));

# Order : Relation, Primary, Secondary
tuples = tuplesData['data'];

# Create instance of the task
task = ImageRetriever();

# word2vec path
embedPath = '/home/satwik/VisualWord2Vec/models/vp_coco_embeddings.bin';
task.readWord2Vec(embedPath);

# Read the ground truth
gtPath = dataPath + 'pilot_gt.txt';
task.readGroundTuples(gtPath);

# Score a tuple
#pickle.dump(task, open('image_retrieval.p', 'wb'));

# Load the pickle
#task = pickle.load(open('image_retrieval.p', 'rb'));
task.performTask(tuples);
