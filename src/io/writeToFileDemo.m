% Script to read the MAT files and write to text files
% We use the following format:
% <PrimaryName, SecondaryName, Relation>:<feature>
%
% First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
% Next clause is read from : /home/satwik/VisualWord2Vec/data/features

dataPath = '/home/satwik/VisualWord2Vec/data'; 
prsFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');
writeToFile(dataPath, prsFeaturePath, numFeaturePath);
