# Script to generate the pilot for odd one out
import pickle
import random
from OddFinder import OddFinder

noTriplets  = 200;

# Read the tuples file
dataPath = '/home/satwik/VisualWord2Vec/data/text-ret/text_ret_final_lemma.p';

with open(dataPath, 'rb') as dataPath:
    tupData = pickle.load(dataPath);

relations = set([j[0] for i in tupData['data'].values() for j in i]);

# Pick out random 4-tuples
#data = [random.sample(relations, 4) for i in xrange(noTriplets)]
#with open('odd-data-4.pickle', 'wb') as dataFile:
#    pickle.dump(data, dataFile);
with open('odd-data-4.pickle', 'rb') as dataFile:
    data = pickle.load(dataFile);

modelPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/';
w2vPath = modelPath + 'al_vectors.txt';
visPath = modelPath + 'word2vec_after_refine_bestmodel.bin';

#finder = OddFinder();
#finder.readEmbeddings(visPath, w2vPath);
#finder.performTask(data);
