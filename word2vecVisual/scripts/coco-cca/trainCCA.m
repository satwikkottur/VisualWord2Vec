% This script reads the image captions and computes the cca cbeforeo-efficients
% for sentences along with feature vectors

dataPath = '/home/satwik/VisualWord2Vec/data/';

% Reading the caption features
trainPath = fullfile(dataPath, 'coco-cca/wiki/train_caption_embeds_after.txt');
mapPath = fullfile(dataPath, 'coco-cca/wiki/test_caption_maps.txt');
testPath = fullfile(dataPath, 'coco-cca/wiki/test_caption_embeds_after.txt');

trainCaptions = dlmread(trainPath, '', 1, 0);
testCaptions = dlmread(testPath, '', 1, 0);
map = dlmread(mapPath);
fprintf('Read textual features...\n');

% Read the visual features
visPath = fullfile(dataPath, 'coco-cnn/fc7_features_train.txt');
visFeats = dlmread(visPath, '', 1, 0);
fprintf('Read visual features...\n');

% Compute the canonical correlation
% Input: visual and textual features
% Output : CCA matrix for visual and textual features along with
%              transformed features
[~, capTrans, ~, ~, textTrainCCA] = canoncorr(visFeats, trainCaptions);

% Forget the visual features (to save space)
clear visFeats;

% Save the variables and load them
save('cca_workspace_wiki_after.mat');
fprintf('Saved the variables...\n');

%load('cca_workspace_variables.mat');
%fprintf('Loaded the variables...\n');

% Transform the test variables into CCA space
textTestCCA = bsxfun(@minus, testCaptions, mean(trainCaptions)) * capTrans;
% Recalls and median recall
[rec, med] = testCCA(textTrainCCA, textTestCCA, map);
rec
med
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing with random numbers
%noTrain = 1000;
%text = rand(noTrain, 20);
%vis = rand(noTrain, 50);
%[A, B, ~, U, V] = canoncorr(vis, text);
%sum(sum(U - bsxfun(@minus, vis, mean(vis))*A))
