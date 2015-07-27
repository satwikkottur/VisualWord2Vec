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
%embeddingType =  3;

% Select true if you want to train the models, false to preload
%trainModel = false;

% Saving the trained models
% Given a name if trained model is to be saved, ignore else
% Models will be save as <name>_P, <name>_R, <name>_S in models/ folder
%saveName = 'fs';

% Verbosity for SVM training
%verboseSVM = false;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%tic
% Setting up the path
%clearvars
%addPaths;
%rootPath = '/home/satwik/VisualWord2Vec/';
%dataPath = '/home/satwik/VisualWord2Vec/data';
%psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
%numFeaturePath = fullfile(dataPath, 'Num_features.txt');

% Model for word2vec embedding
%switch embeddingType
%    case 1
%        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v.mat'); 
%    case 2
%        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_tokenized.mat'); 
%    case 3
%        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_tokenized_stops.mat'); 
%    case 4
%        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_raw.mat');
%
%    case 5
%        word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_fs.mat');
%end
%
%% Reading the labels
%[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);

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
if ~trainModel
    switch embeddingType
        case 1
            load(fullfile(rootPath, 'models', 'w_model_coco.mat'));

        case 2
            load(fullfile(rootPath, 'models', 'w_model_coco_tokenized.mat'));

        case 3
            load(fullfile(rootPath, 'models', 'w_model_coco_tokenized_stops.mat'));

        case 4
            load(fullfile(rootPath, 'models', 'w_model_coco_raw.mat'));
            %load(fullfile(rootPath, 'models', 'w_model_coco_raw_orig.mat'));

        case 5
            load(fullfile(rootPath, 'models', 'w_model_coco_fs.mat'));
    end
    %load(fullfile(rootPath, 'models', 'workspacedump_w_models_coco.mat'));
else
    % Cross validations
    noFolds = 5;
    % Smaller sweep of c
    cRange = [0.001, 0.01, 0.1, 1.0];
    %cRange = [0.01];
    %cRange = [0.0001, 0.0005, 0.001, 0.005, 0.01, 005, 0.1, 0.5, 1.0, 5.0, 10.0];
    noNegatives = 12780;
    rndSeed = 100;

    % Train models for P, S, R
    [R_model_test_embed R_model_crossval_embed R_acc_crossval_embed R_random_crossval_embed] = ...
                    embedding(Rencoding, Rembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', ['model_variables_coco_', saveName, '_', num2str(noInst), '_R.mat']), ...
                                'R_model_test_embed', 'R_acc_crossval_embed');

    [P_model_test_embed P_model_crossval_embed P_acc_crossval_embed P_random_crossval_embed] = ...
                    embedding(Pencoding, Pembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', ['model_variables_coco_', saveName, '_', num2str(noInst), '_P.mat']), ...
                                'P_model_test_embed', 'P_acc_crossval_embed');

    [S_model_test_embed S_model_crossval_embed S_acc_crossval_embed S_random_crossval_embed] = ...
                    embedding(Sencoding, Sembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', ['model_variables_coco_', saveName, '_', num2str(noInst), '_S.mat']), ...
                                'S_model_test_embed', 'S_acc_crossval_embed');

    % Find the best C based on *_acc_crossval_embed (Here I'm fixing to 0.01), and turn w into matrix for efficient score computation
    bestInd = 2;
    R_A = reshape(R_model_test_embed{bestInd}.w, [], 200);
    P_A = reshape(P_model_test_embed{bestInd}.w, [], 200);
    S_A = reshape(S_model_test_embed{bestInd}.w, [], 200);
    
    % Dumping  variables
    fprintf('Dumping model variables for iteration\n');
    save(fullfile(rootPath, 'models', ['w_model_coco_', saveName, '_', num2str(noInst), '.mat']));
    fprintf('Models dumped after sweeping c\n');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
