% Wrapper for showing the usage of clustering of relationship words

% Reading the relationship features from the features file
tic

run('../addPaths');
rootPath = '/home/satwik/VisualWord2Vec/';
dataPath = '/home/satwik/VisualWord2Vec/data';

% Reading the features and relations
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');

[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);

% Assigning the cluster ids (each relation word is a cluster)
[]


% Visualization

% Clustering the features


% Assigning the cluster ids after clustering

% Visualization

time = toc;
fprintf('Total time taken : %f\n', time);
