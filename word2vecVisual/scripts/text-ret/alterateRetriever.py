# Script to read the tuples for the text based image retrieval task
import pickle
from ImageRetriever import ImageRetriever
import multiprocessing

print 'Started the task'
# Setting up paths
dataPath = '/home/satwik/VisualWord2Vec/data/text-ret/';
tupPath = dataPath + 'text_ret_final_lemma.p';
with open(tupPath, 'rb') as dataFile:
    tupData = pickle.load(dataFile);

# Create instance of the task
task = ImageRetriever('SINGLE');

# Read the ground truth
gtPath = dataPath + 'pilot_gt.txt';
task.readGroundTuples(gtPath);

# Reading embeddings for multiple models
embedPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/al_vectors.txt';
#embedPath = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/wiki_iters/word2vec_wiki_iter_0.bin";
task.loadWord2Vec(embedPath, 'raw');

embedPath = '/home/satwik/VisualWord2Vec/data/word2vec_output_bestmodel_single.bin';
#embedPath = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/wiki_iters/word2vec_wiki_iter_24.bin";
task.loadWord2Vec(embedPath);

# Vocab file for coco
vocabPath = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/refineVocab_coco.bin";
# Load the refined vocab (words that got refined)
task.loadRefineVocab(vocabPath);

# Perform the task
task.performTask(tupData['data']);

# Setup the mutliprocessing framework
'''task.setupTrainTestCOCO(captions);

noThreads = 32;
task.setupMultiProcessing(noThreads);

# Perform the task on COCO (multi)
jobs = [];
for i in xrange(0, noThreads):
    p = multiprocessing.Process(target=task.performTaskMultiCOCO, args = (i, noThreads));
    jobs.append(p);
    p.start();
#task.performTaskCOCO(captions);'''

