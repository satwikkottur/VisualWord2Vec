% Script to get limited training data from the whole of training data
% This is based on R features

rootPath = '/home/satwik/VisualWord2Vec/';
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'float_features_withoutheader.txt');
addpath(genpath(fullfile(rootPath, 'code/io')));

% Reading the labels
[PlabelAll, SlabelAll, RlabelAll, RfeaturesAll] = readFromFile(psrFeaturePath, numFeaturePath);
[uniqR, ~, uniqInds] = unique(RlabelAll);

prsTemplate = fullfile(dataPath, 'PSR_features_%d.txt');
numTemplate = fullfile(dataPath, 'float_features_%d.txt');
% Choosing a part of the data
% Number of instances per relation 2 : 2 : 20
for noInst = [1, 5, 10, 15, 20]
    trainIndices = [];
    fprintf('Current iteration : %d....\n\n\n', noInst);
    for indR = 1:length(uniqR)
        trainIndices = [trainIndices; datasample(find(uniqInds == indR), noInst)];
    end

    % Shuffling the training indices to mimic randomness in data (needed?)
    trainIndices = trainIndices(randperm(length(trainIndices)));

    % Taking the relevant P, R, S, visual features
    Plabel = PlabelAll(trainIndices);
    Rlabel = RlabelAll(trainIndices);
    Slabel = SlabelAll(trainIndices);
    Rfeatures = RfeaturesAll(trainIndices, :);

    % Save the files in data
    prsName = sprintf(prsTemplate, noInst);
    numName = sprintf(numTemplate, noInst);
    
    saveToFile(prsName, numName, Plabel, Rlabel, Slabel, Rfeatures);
end
