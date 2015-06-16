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
[Pencoding, Pdict, Plabels] = oneHotEncode(Plabel);
[Sencoding, Sdict, Slabels] = oneHotEncode(Slabel);
[Rencoding, Rdict, Rlabels] = oneHotEncode(Rlabel);

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
debugValTestSets;

toc
