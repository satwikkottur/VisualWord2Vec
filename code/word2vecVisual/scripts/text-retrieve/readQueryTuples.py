# Script to read the tuples for the text based image retrieval task
import pickle
from ImageRetriever import ImageRetriever

print 'Started the task'
# Setting up paths
dataPath = '/home/satwik/VisualWord2Vec/data/text-retrieval/';
picklePath = dataPath + 'text_retrieval_pilot_lemma.p';
tuplesData = pickle.load(open(picklePath, 'rb'));

# Order : Relation, Primary, Secondary
tuples = tuplesData['data'];

# Create instance of the task
task = ImageRetriever();

# Read the ground truth
gtPath = dataPath + 'pilot_gt.txt';
task.readGroundTuples(gtPath);

#embedPath = '/home/satwik/VisualWord2Vec/models/vp_coco_embeddings.bin';
# word2vec path 0.156923076923 0.391538461538 0.510769230769
'''embedPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_before_refine.bin';
task.readWord2Vec(embedPath);
# Perform the task
task.performTask(tuples);

# Read after refining and perform task again 0.166923 0.3853846 0.5330769 
embedPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_after_refine.bin';
task.readWord2Vec(embedPath);
# Perform the task
task.performTask(tuples);

#0.147692307692 0.362307692308 0.483076923077
embedPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/al_vectors.txt';
task.readWord2Vec(embedPath);
# Perform the task
task.performTask(tuples);'''

#embedPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_output_bestmodel_single.bin';
embedPath = '/home/satwik/VisualWord2Vec/models/wiki_embeddings_pre_refine.bin';
#0.133076923077 0.324615384615 0.422307692308
task.readWord2Vec(embedPath);
# Perform the task
task.performTask(tuples);

#0.124615384615 0.290769230769 0.383076923077
embedPath = '/home/satwik/VisualWord2Vec/models/wiki_embeddings_post_refine.bin';
task.readWord2Vec(embedPath);
# Perform the task
task.performTask(tuples);
