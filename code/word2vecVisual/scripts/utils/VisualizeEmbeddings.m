% Script to visualize the embeddings of the words before and after the training

run('/home/satwik/VisualWord2Vec/code/addPaths.m');
rootDir = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/';

% Pre-training and post training
prePath = fullfile(rootDir, 'word2vec_pre.txt');
postPath = fullfile(rootDir, 'word2vec_post.txt');
vocabPath = fullfile(rootDir, 'word2vec_vocab.txt');

% Reading the features files
preFeatures = dlmread(prePath);
postFeatures = dlmread(postPath);

% Reading the vocabulary
%fileId = fopen(vocabPath, 'r');
%vocab = fread(fileId, '*char');

%vocab = regexp(vocab, '.*\n', 'match');

% Visualize before and after vectors
noDims = 2;
noInitDims = 30;
perplexity = 3;
tsnePre = tsne(preFeatures, [], noDims, noInitDims, perplexity);
tsnePost = tsne(postFeatures, [], noDims, noInitDims, perplexity);

figure(1); scatter(tsnePre(:, 1), tsnePre(:, 2));
figure(2); scatter(tsnePost(:, 1), tsnePost(:, 2));
