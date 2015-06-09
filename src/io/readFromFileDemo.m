% Script to read features from a text file
% We use the following format:
% <PrimaryName, SecondaryName, Relation>:<feature>
%
% First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
% Next clause is read from : /home/satwik/VisualWord2Vec/data/features

% Setting up the path
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');

[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);
