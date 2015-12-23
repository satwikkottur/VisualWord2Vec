% Script to get limited training data from the whole of training data
% This is based on R features

rootPath = '/home/satwik/VisualWord2Vec/';
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'float_features_withoutheader.txt');
addpath(genpath(fullfile(rootPath, 'code/io')));

% Reading the labels
[PlabelAll, SlabelAll, RlabelAll, RfeaturesAll] = ...
                            readFromFile(psrFeaturePath, numFeaturePath);
[uniqR, ~, uniqInds] = unique(RlabelAll);

prsTemplate = fullfile(dataPath, 'PSR_features_R_%d.txt');
numTemplate = fullfile(dataPath, 'float_features_R_%d.txt');
% Choosing a part of the data
% Number of R, 20:200
%for noR = 20:20:200
for noR = [120]
    % Choose the R's that need to be picked
    selectR = datasample(uniqR, noR, 'Replace', false);

    trainIndices = [];
    fprintf('Current iteration : %d....\n\n', noR);
    for indR = 1:noR
        trainIndices = [trainIndices; ...
                        find(strcmp(RlabelAll, uniqR{indR}) == 1)];
    end

    % Shuffling the training indices to mimic randomness in data (needed?)
    trainIndices = trainIndices(randperm(length(trainIndices)));

    % Taking the relevant P, R, S, visual features
    Plabel = PlabelAll(trainIndices);
    Rlabel = RlabelAll(trainIndices);
    Slabel = SlabelAll(trainIndices);
    Rfeatures = RfeaturesAll(trainIndices, :);

    % Save the files in data
    prsName = sprintf(prsTemplate, noR);
    numName = sprintf(numTemplate, noR);
    
    saveToFile(prsName, numName, Plabel, Rlabel, Slabel, Rfeatures);
end
