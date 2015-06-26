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

tic
% Setting up the path
%clearvars
addPaths;
rootPath = '/home/satwik/VisualWord2Vec/';
dataPath = '/home/satwik/VisualWord2Vec/data';
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');

% Model for word2vec embedding
word2vecModel = fullfile(rootPath, 'models', 'coco_w2v_tokenized.mat'); 
%word2vecModel = fullfile(rootPath, 'models', 'coco_tokenized_stops_word2vec.mat'); 
%word2vecModel = fullfile(rootPath, 'models', 'coco_w2v.mat'); 

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
if 0
    %load(fullfile(rootPath, 'models', 'workspacedump_w_models_coco.mat'));
    load(fullfile(rootPath, 'models', 'models_coco_tokenized.mat'));
else
    % Cross validations
    noFolds = 5;
    cRange = [0.0001, 0.0005, 0.001, 0.005, 0.01, 005, 0.1, 0.5, 1.0, 5.0, 10.0];
    noNegatives = 12780;
    rndSeed = 100;

    % Train models for P, S, R
    [R_model_test_embed R_model_crossval_embed R_acc_crossval_embed R_random_crossval_embed] = ...
                    embedding(Rencoding, Rembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', 'model_variables_coco_tokenized_R.mat'), ...
                                'R_model_test_embed', 'R_acc_crossval_embed');

    [P_model_test_embed P_model_crossval_embed P_acc_crossval_embed P_random_crossval_embed] = ...
                    embedding(Pencoding, Pembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', 'model_variables_coco_tokenized_P.mat'), ...
                                'P_model_test_embed', 'P_acc_crossval_embed');

    [S_model_test_embed S_model_crossval_embed S_acc_crossval_embed S_random_crossval_embed] = ...
                    embedding(Sencoding, Sembed, Rfeatures, cRange, noFolds, noNegatives, rndSeed);

    save(fullfile(rootPath, 'models', 'model_variables_coco_tokenized_S.mat'), ...
                                'S_model_test_embed', 'S_acc_crossval_embed');

    % Find the best C based on *_acc_crossval_embed (Here I'm fixing to 0.01), and turn w into matrix for efficient score computation
    %R_A = reshape(R_model_test_embed{3}.w, [ndims,200]);
    %P_A = reshape(P_model_test_embed{3}.w, [ndims,200]);
    %S_A = reshape(S_model_test_embed{3}.w, [ndims,200]);
    % Dumping  variables
    fprintf('Dumping model variables\n');
    %save(fullfile(rootPath, 'models', 'model_variables_coco_tokenized_stops.mat'));
    fprintf('Models dumped after sweeping c\n');
end
return
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
testRscore = Rscore * testRembed';
testSscore = Sscore * testSembed';
testPscore = Pscore * testPembed';

% Compute x'Ay for each x, y in val
valRscore = Rscore * valRembed';
valPscore = Pscore * valPembed';
valSscore = Sscore * valSembed';

% Manually change threshold until precVal is maximized.
% Visual threshold usually around 0-1
threshold = 0.6; % Empirically determined

%precValues = [];
%for threshold = 0:0.05:1.5
%    % Faster implementation
%    visualValScore = mean(max(...
%                        valRscore(:, valRlabels) + ...
%                        valPscore(:, valPlabels) + ...
%                        valSscore(:, valSlabels) - ...
%                        threshold, 0), 1)';
%    [precVal, baseVal] = precision(visualValScore, valLabel);
%
%    fprintf('Val(visual) %f : %f\n', threshold, mean(precVal(:)))
%    precValues = [precValues, mean(precVal(:))];
%end

% Now check the performance on test dataset with this threshold
% Faster implementation
visualTestScore = mean(max(...
                        testRscore(:, testRlabels) + ...
                        testPscore(:, testPlabels) + ...
                        testSscore(:, testSlabels) - ...
                        threshold, 0), 1)';
[precTest, baseTest] = precision(visualTestScore, testLabel);
fprintf('Test (visual) : %f\n', mean(precTest(:)));
return

% Debugging the visual features
% debugVisualFeatures;
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
%threshold = -1.2;
threshold = -1.4;

% Faster implementation
textValScore = mean(max(...
                        valRscoreText(:, valRlabels) + ...
                        valPscoreText(:, valPlabels) + ...
                        valSscoreText(:, valSlabels) - ...
                        threshold, 0), 1)';
[precVal, baseVal] = precision(textValScore, valLabel);

% Use the same threshold for test set
% Faster implementation
textTestScore = mean(max(...
                    testRscoreText(:, testRlabels) + ...
                    testPscoreText(:, testPlabels) + ...
                    testSscoreText(:, testSlabels) - ...
                    threshold, 0), 1)';
[precTest, baseTest] = precision(textTestScore, testLabel);
fprintf('Test (textual) : %f\n', mean(precTest(:)));

% Debugging the textual features
%debugTextualFeatures;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%% Visual + Text features %%%%%%%%%%%%%%%%%%%%
valHybridFeatures = [textValScore, visualValScore];
testHybridFeatures = [textTestScore, visualTestScore];

% Fine tune c until optimal is obtained
%c = 10000;
c = 1000;
noFolds = 5;
verbose = false;

% Cross validation
[hybridModelTest, hybridModelCrossval, hybridAccCrossval, hybridRandomCrossval] = ...
            perclass(valLabel * 2 - 1, valHybridFeatures, c, noFolds, verbose);            
hybridPerfCrossval = mean(hybridAccCrossval);

% Testing
[~, ~, hybridScoreTest] = predict(testLabel * 2 - 1, sparse(testHybridFeatures), hybridModelTest{1});
[hybridPerfTest, baselineTest] = precision(hybridScoreTest, testLabel * 2 - 1);
fprintf('Test (visual+textual) : %f\n', mean(hybridPerfTest(:)));
corr(hybridScoreTest, testScore, 'type', 'Spearman');
corr(hybridScoreTest, testScore, 'type', 'Kendall');

% Debugging the visual + textual features
%debugVisualTextFeatures;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%% Bing baseline %%%%%%%%%%%%%%%%%%%%%%%
bingVal = load('/home/satwik/VisualWord2Vec/models/bing_val.mat');
bingVal = double(bingVal.data);
bingTest = load('/home/satwik/VisualWord2Vec/models/bing_test.mat');
bingTest = double(bingTest.data);

% Fine tune c until optimum is obtained
c = 0.0001;
noFolds = 5;

% Cross validation
[bing_model_test bing_model_crossval bing_acc_crossval bing_random_crossval] = ...
                            perclass(valLabel * 2 - 1,log(bingVal+1), c, noFolds, verbose);
bing_perf_crossval = mean(bing_acc_crossval);

%test
[~, ~, bing_score_test] = predict(testLabel * 2 - 1, sparse(log(bingTest+1)), bing_model_test{1});
[bing_perf_test,baseline_test] = precision(bing_score_test, testLabel * 2 - 1);
corr(bing_score_test, testScore, 'type', 'Spearman');
corr(bing_score_test, testScore, 'type', 'Kendall');

%Get val score of bing, potentially for hybrid models
[~, ~, bing_score_val] = predict(valLabel * 2 - 1, sparse(log(bingVal+1)), bing_model_test{1});

% Debugging the visual + textual features
%debugBingFeatures;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
toc
