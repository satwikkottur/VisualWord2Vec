% This script reads the image captions and computes the cca co-efficients
% for sentences along with feature vectors

dataPath = '/home/satwik/VisualWord2Vec/data/';

% Reading the caption features
trainPath = fullfile(dataPath, 'coco-cca/train_caption_embeds.txt');
mapPath = fullfile(dataPath, 'coco-cca/test_caption_maps.txt');
testPath = fullfile(dataPath, 'coco-cca/test_caption_embeds.txt');

trainCaps = dlmread(trainPath, '', 1, 0);
testCaps = dlmread(testPath, '', 1, 0);
map = dlmread(mapPath);
fprintf('Read textual features...\n');

% Setting the local aliases
noTrain = length(trainCaps);
noTest = length(testCaps);
capFeatSize = size(trainCaps, 2);

% Read the visual features
visPath = fullfile(dataPath, 'coco-cnn/fc7_features_train.txt');
visFeats = dlmread(visPath, '', 1, 0);
fprintf('Read visual features...\n');

% Compute the canonical correlation
% Input: visual and textual features
% Output : CCA matrix for visual and textual features along with
%              transformed features
[visTrans, capTrans, ~, visCCA, textTrainCCA] = canoncorr(visFeats, trainCaps);
%[A, B, ~, U, V] = canoncorr(vis, text);

% Save the variables and load them
save('cca_workspace_variables.mat');
fprintf('Saved the variables...\n');
