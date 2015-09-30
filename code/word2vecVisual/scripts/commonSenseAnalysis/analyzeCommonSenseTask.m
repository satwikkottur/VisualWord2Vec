% Script that reads the dumps from C code and does qualitative analysis 
% for common sense task

% Setting up paths
rootPath = '/home/VisualWord2Vec/code/word2vecVisual';

preFile = fullfile(rootPath, 'modelsNdata/word2vec_pre_0_0_1_25.txt');
postFile = fullfile(rootPath, 'modelsNdata/word2vec_post_0_0_1_25.txt');
vocabFile = fullfile(rootPath, 'modelsNdata/word2vec_vocab_0_0_1_25.txt');

% Read the embeddings before / after refining along with vocab
[preEmbed, postEmbed, featureWords] = ...
                        readEmbedFile(preFile, postFile, vocabFile, true);

% Reading the tuples that caused an increase
readImprovedTuples();

% Sort the improved tuples in order and pick max words

% Embed through t-sne


% Read
