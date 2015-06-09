% Script to learn the model for the json data
% Assumes the same flow of learn_model.m of the original source code
%
% Following changes are made from learn_model.m
% 1. Uses a .txt file to read <primary, secondary, relation>:feature instead of mat files for faster access
% 2. One hot encoding based on the unique labels for all P,R,S/
% 3. 

% Reading the features from the files
% First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
% Next clause is read from : /home/satwik/VisualWord2Vec/data/features

tic
% Setting up the path
%clearvars
addPaths;
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');

% Reading the labels
[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);

% Debugging reading files
% debugReadingFiles;

% Number of images
noImages = size(Rfeatures, 1);

% Encoding P, R, S labels using one hot encoding(unique labels)
[Pencoding, Pdict] = oneHotEncode(Plabel);
[Sencoding, Sdict] = oneHotEncode(Slabel);
[Rencoding, Rdict] = oneHotEncode(Rlabel);

Rfeatures = double(Rfeatures);

% Debugging encoding
%debugEncoding;
toc
