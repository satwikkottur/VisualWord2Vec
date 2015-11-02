# Script to test the embeddings qualitative results
import sys

# Load the embeddings (before and after)
rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/';
sys.path.append(rootPath + 'scripts/utils/')
from loadWord2Vec import loadWord2Vec

beforePath = rootPath + 'modelsNdata/word2vec_wiki_iter_0.bin';
before = loadWord2Vec(beforePath);

afterPath = rootPath + 'modelsNdata/word2vec_wiki_iter_24.bin';
after = loadWord2Vec(afterPath);

