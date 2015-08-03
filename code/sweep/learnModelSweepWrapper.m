% Script to learn the model for the json data
% Assumes the same flow of learn_model.m of the original source code
%
% Following changes are made from learn_model.m
% 1. Uses a .txt file to read <primary, secondary, relation>:feature instead of mat files for faster access
% 2. One hot encoding based on the unique labels for all P,R,S/
% 3. Computing the word embedding using a trained model
% 4. Training the models for P,S,R words
% 5. Evaluation for validation and testing sets. Fixing thresholds using validation set
% 6. Considering all the cases:
%       a. Visual features
%       b. Textual features
%       c. Textual + Visual features
%       d. Bing baselines
%

% Reading the features from the files
% First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
% Next clause is read from : /home/satwik/VisualWord2Vec/data/features

%%%%%%% Parameters and flags %%%%%%%%
% Select the embedding type
% 1 - plain coco
% 2 - coco with tokenization, sentences separated by space
% 3 - coco with tokenization, sentences separated by newline
% 4 - coco with word2vec trained on raw text
% 5 - coco with tokeniation, sentences separated by full stop
embeddingType =  1;

% Select true if you want to train the models, false to preload
trainModel = true;

% Saving the trained models
% Given a name if trained model is to be saved, ignore else
% Models will be save as <name>_P, <name>_R, <name>_S in models/ folder
saveName = 'sweep';

% Verbosity for SVM training
verboseSVM = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic
% Setting up the path
%clearvars
addPaths;
rootPath = '/home/satwik/VisualWord2Vec/';
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');

% Model for word2vec embedding
switch embeddingType
    case 1
        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v.mat'); 
    case 2
        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_tokenized.mat'); 
    case 3
        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_tokenized_stops.mat'); 
    case 4
        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_raw.mat');
    case 5
        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_fs.mat');
end

% Reading the labels
[PlabelAll, SlabelAll, RlabelAll, RfeaturesAll] = readFromFile(psrFeaturePath, numFeaturePath);
[uniqR, ~, uniqInds] = unique(RlabelAll);

% Choosing a part of the data
% Number of instances per relation 2 : 2 : 20
for noInst = 20:2:20
%for noInst = 10:2:10
%for noInst = 2:2:20
    trainIndices = [];
    fprintf('Current iteration : %d....\n\n\n', noInst);
    for indR = 1:length(uniqR)
        trainIndices = [trainIndices; datasample(find(uniqInds == indR), noInst)];
    end

    % Shuffling the training indices to mimic randomness in data (needed?)
    %trainIndices = trainIndices(randperm(1:length(trainIndices)));

    Plabel = PlabelAll(trainIndices);
    Rlabel = RlabelAll(trainIndices);
    Slabel = SlabelAll(trainIndices);

    Rfeatures = RfeaturesAll(trainIndices, :);

    %size(Plabel)
    %size(Rlabel)
    %size(Slabel)
    %size(Rfeatures)
    %size(unique(Rlabel))

    learnModelSweep
    fprintf('=========\n\n');
end
    learnModelEvaluation
