% Script to learn the model for the json data
% Assumes the same flow of learn_model.m of the original source code
%
% Following changes are made from learn_model.m
% 1. Uses a .txt file to read <primary, secondary, relation>:feature instead of mat files for faster access
% 2. One hot encoding based on the unique labels for all P,R,S/
% 3. Computing the word embedding using a trained model
% 4. Training the models for P,S,R words

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
word2vecModel = '/home/satwik/VisualWord2Vec/models/coco_w2v.mat'; % Model for word2vec embedding

% Reading the labels
[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);

% Debugging reading files
% debugReadingFiles;

% Number of images
noImages = size(Rfeatures, 1);

% Encoding P, R, S labels using one hot encoding(unique labels)
[Pencoding, Pdict, Pinds] = oneHotEncode(Plabel);
[Sencoding, Sdict, Sinds] = oneHotEncode(Slabel);
[Rencoding, Rdict, Rinds] = oneHotEncode(Rlabel);

Rfeatures = double(Rfeatures);

% Debugging encoding
% debugEncoding;

% Word2vec embedding using the model for the dictionary
w2vModel = load(word2vecModel);
Pembed = embedLabels(Pdict, w2vModel);
Sembed = embedLabels(Sdict, w2vModel);
Rembed = embedLabels(Rdict, w2vModel);

% Debugging the word2vec embedding
%debugEmbedding;

%%%%%%%%%%%%%%%%%%%% Original code %%%%%%%%%%%%%%%%%%
%TODO: don't preload when you want to train your own model.
if 1
    %load workspacedump_w_models_coco.mat;
else
    % Cross validations
    noFolds = 5;
    cRange = [0.0001, 0.001, 0.01, 0.1];
    noNegatives = 12780;
    rndSeed = 100;

    % Train models for P, S, R
    [R_model_test_embed R_model_crossval_embed R_acc_crossval_embed R_random_crossval_embed] = ...
                    embedding(Rencoding, Rembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    [P_model_test_embed P_model_crossval_embed P_acc_crossval_embed P_random_crossval_embed] = ...
                    embedding(Pencoding, Pembed, Pfeatures, cRange, noFolds, noNegatives, rndSeed);

    [S_model_test_embed S_model_crossval_embed S_acc_crossval_embed S_random_crossval_embed] = ...
                    embedding(Sencoding, Sembed, Sfeatures, cRange, noFolds, noNegatives, rndSeed);

    % Find the best C based on *_acc_crossval_embed (Here I'm fixing to 0.01), and turn w into matrix for efficient score computation
    R_A = reshape(R_model_test_embed{3}.w, [ndims,200]);
    P_A = reshape(P_model_test_embed{3}.w, [ndims,200]);
    S_A = reshape(S_model_test_embed{3}.w, [ndims,200]);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load validation and test data
val = load('/home/satwik/VisualWord2Vec/original/val.mat');
test = load('/home/satwik/VisualWord2Vec/original/test.mat');

noTest = size(test.data, 1);
noVal = size(val.data, 1);

% Cleaning strings
[valP, valR, valS] = cleanStrings(val.data);
[testP, testR, testS] = cleanStrings(test.data);

% Index and get word2vec embedding for val and test PRS
% Validate set P, R, S
[~, valRdict, valRlabels] = oneHotEncode(valR);
valRembed = embedLabels(valRdict, w2vModel);

[~, valPdict, valPlabels] = oneHotEncode(valP);
valPembed = embedLabels(valPdict, w2vModel);

[~, valSdict, valSlabels] = oneHotEncode(valS);
valSembed = embedLabels(valSdict, w2vModel);

% Test set P, R, S
[~, testRdict, testRlabels] = oneHotEncode(testR);
testRembed = embedLabels(testRdict, w2vModel);

[~, testPdict, testPlabels] = oneHotEncode(testP);
testPembed = embedLabels(testPdict, w2vModel);

[~, testSdict, testSlabels] = oneHotEncode(testS);
testSembed = embedLabels(testSdict, w2vModel);

% Debugging validation and training set cleaning
%debugValTestSets;

% Computing human scores for the labels
valScore = cell2mat(val.data(:, 3));
valLabel = valScore > 0;

testScore = cell2mat(test.data(:, 3)) > 0;
testLabel = testScore > 0 ;

%%%%%%%%%%%%%%%%% Visual Model %%%%%%%%%%%%%%%%%%%
% Compute x'A for each x
Rscore = Rfeatures * R_A;
Pscore = Rfeatures * P_A;
Sscore = Rfeatures * S_A;

% Compute x'Ay for each x, y in test
testRscore = Rscore * testRembed;
testSscore = Sscore * testSembed;
testPscore = Pscore * testPembed;

% Compute x'Ay for each x, y in val
valRscore = Rscore * valRembed;
valPscore = Pscore * valPembed;
valSscore = Sscore * valSembed;

% Manually change threshold until precVal is maximized.
% Visual threshold usually around 0-1
threshold = 0.6; % Empirically determined

visualValScore = zeros(noVal, 1);
for i = 1:noVal
    visualValScore = mean(max(...
                            valRscore(:, valRlabels(i)) + ...
                            valPscore(:, valPlabels(i)) + ...
                            valSscore(:, valSlabels(i)) - ...
                            threshold, 0), 1);
end
[precVal, baseVal] = precision(visualValScore, valLabel);

% Now check the performance on test dataset with this threshold
visualTestScore = zeros(noTest, 1);
for i = 1:noTest
    visualTestScore = mean(max(...
                            testRscore(:, testRlabels(i)) + ...
                            testPscore(:, testPlabels(i)) + ...
                            testSscore(:, testSlabels(i)) - ...
                            threshold, 0), 1);
end
[precTest, baseTest] = precision(visualTestScore, testLabel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% Only Text features %%%%%%%%%%%%%%%%%%%%%%%%%
% Compute cosine similarity - 1 for val 
valRscoreText = - pdist2(Rembed(Rinds, :), valRembed, 'cosine');
valPscoreText = - pdist2(Pembed(Pinds, :), valPembed, 'cosine');
valSscoreText = - pdist2(Sembed(Sinds, :), valSembed, 'cosine');

% Compute cosine similarity - 1 for test
testRscoreText = - pdist2(Rembed(Rinds, :), testRembed, 'cosine');
testPscoreText = - pdist2(Pembed(Pinds, :), testPembed, 'cosine');
testSscoreText = - pdist2(Sembed(Sinds, :), testSembed, 'cosine');

% Manually adjust the threshold until precVal is maximized
% Threshold around -2 ~ 1
threshold = -1.2;

textValScore = zeros(noVal, 1);
for i = 1:noVal
    textValScore = mean(max(...
                            valRscoreText(:, valRlabels(i)) + ...
                            valPscoreText(:, valPlabels(i)) + ...
                            valSscoreText(:, valSlabels(i)) - ...
                            threshold, 0), 1);
end
[precVal, baseVal] = precision(valScoreText, valLabel);

% Use the same threshold for test set
textTestScore = zeros(noTest, 1);
for i = 1:noTest
    textTestScore = mean(max(...
                            testRscoreText(:, testRlabels(i)) + ...
                            testPscoreText(:, testPlabels(i)) + ...
                            testSscoreText(:, testSlabels(i)) - ...
                            threshold, 0), 1);
end
[precTest, baseTest] = precision(testScoreText, testLabel);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% Visual + Text features %%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc
