% This script computes the vocabulary for VP training data
% Also compared it with visual word2vec vocab

% Load the mat dump file
addpath('/home/satwik/VisualWord2Vec/code/word2vecVisual/scripts/');
rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/';
%matDumpPath = fullfile(rootPath, 'vp-fitb/code/dumps/trainWord2Vec.mat');
%load(matDumpPath)
lemmaPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/vp-fitb/code/dumps/vp_train_tokens_lemma.txt';
lemmaFile = fopen(lemmaPath, 'rb');
tokens = textscan(lemmaFile, '%s', 'Delimiter', '\n');
tokens = unique(tokens{1});
fclose(lemmaFile);

% Read the vocab for visualword2vec, close the file
vTokenPath = fullfile(rootPath, 'modelsNdata/split_vocab_vw2v.txt');
vTokenFile = fopen(vTokenPath, 'rb');
vTokens = textscan(vTokenFile, '%s', 'Delimiter', '\n');
fclose(vTokenFile);

% Taken the meaningful element (first)
vTokens = unique(vTokens{1});

% Compare vTokens and tokens
% vTokens = p,r,s training dataset (visual word2vec)
% tokens = VP training dataset

% Common tokens from both the set
members = ismember(tokens, vTokens);
common = tokens(members);
exclude = tokens(~members);

fprintf('Tokens have : %d\nCommon : %d\nTokens not present : %d\n', ...
                        length(tokens), sum(members), sum(1-members));
