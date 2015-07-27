% Script to perform evaluation using the models learnt from
% sweeping the amount of training data for learning the 
% model from abstract clipart images

% Models for each of the sweep (2,4,6,...,18,20 / relation) 
% can be found in the 'models/' folder under the name
% 'w_model_coco_fs_x.mat' where x is the number of data / relation

% Adding the corresponding paths
addPaths;

% Word2Vec models
rootPath = '/home/satwik/VisualWord2Vec/';
word2vecModel = fullfile(rootPath, 'models', 'coco_w2v.mat');
w2vModel = load(word2vecModel);

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

valScore = cell2mat(val.data(:, 3));
valLabel = valScore > 0;

testScore = cell2mat(test.data(:, 3)) > 0;
testLabel = testScore > 0 ;

% Model evaluation for each model
for noInst = 10
%for noInst = 2:2:20
    fprintf('Current iterations: %d\n', noInst);
    % Loading the corresponding model
    modelName = sprintf('/home/satwik/VisualWord2Vec/models/w_model_coco_sweep_%d.mat', noInst);
    %modelName = sprintf('/home/satwik/VisualWord2Vec/models/w_model_coco_fs_%d.mat', noInst);

    load(modelName)
    
    % Opening the file
    fileId = fopen(sprintf('src/dumps/sweep_output_%d', noInst), 'wb');
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
    switch embeddingType
        case 1
            threshold = 0.6; % Empirically determined
        case 2
            threshold = 0.6; % Empirically determined
        case 3
            threshold = 0.5; % Empirically determined
        case 4
            threshold = 0.4; % Empirically determined
            %threshold = 0.6; % Empirically determined
        case 5
            threshold = 0.3; % Empirically determined
    end

    precValues = [];
    for threshold = 0:0.05:1.5
        % Faster implementation
        visualValScore = mean(max(...
                            valRscore(:, valRlabels) + ...
                            valPscore(:, valPlabels) + ...
                            valSscore(:, valSlabels) - ...
                            threshold, 0), 1)';
        [precVal, baseVal] = precision(visualValScore, valLabel);
    %
        fprintf(fileId, 'Val(visual) %f : %f\n', threshold, mean(precVal(:)))
        pecValues = [precValues, mean(precVal(:))];

    % Now check the performance on test dataset with this threshold
    % Faster implementation
    visualTestScore = mean(max(...
                            testRscore(:, testRlabels) + ...
                            testPscore(:, testPlabels) + ...
                            testSscore(:, testSlabels) - ...
                            threshold, 0), 1)';
    [precTest, baseTest] = precision(visualTestScore, testLabel);
    fprintf(fileId, 'Test (visual) : %f\n', mean(precTest(:)));
    end

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
    switch embeddingType
        case 1
            threshold = -1.2; % Empirically determined
        case 2
            threshold = -1.4; % Empirically determined
        case 3
            threshold = -1.4; % Empirically determined
        case 4
            threshold = -1.3; % Empirically determined
            %threshold = -1.2; % Empirically determined
        case 5
            threshold = -1.4; % Empirically determined

    end

    %precValues = [];
    % Faster implementation
    for threshold = -2:0.1:1
        textValScore = mean(max(...
                                valRscoreText(:, valRlabels) + ...
                                valPscoreText(:, valPlabels) + ...
                                valSscoreText(:, valSlabels) - ...
                                threshold, 0), 1)';
        [precVal, baseVal] = precision(textValScore, valLabel);
        fprintf(fileId, 'Test (textual) %f : %f\n', threshold, mean(precVal(:)));

        precValues = [precValues, mean(precVal(:))];
    %end

    % Use the same threshold for test set
    % Faster implementation
    textTestScore = mean(max(...
                        testRscoreText(:, testRlabels) + ...
                        testPscoreText(:, testPlabels) + ...
                        testSscoreText(:, testSlabels) - ...
                        threshold, 0), 1)';
    [precTest, baseTest] = precision(textTestScore, testLabel);
    fprintf(fileId, 'Test (textual) : %f\n', mean(precTest(:)));
    end

    % Debugging the textual features
    %debugTextualFeatures;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    %%%%%%%%%%%%% Visual + Text features %%%%%%%%%%%%%%%%%%%%
    valHybridFeatures = [textValScore, visualValScore];
    testHybridFeatures = [textTestScore, visualTestScore];

    % Fine tune c until optimal is obtained
    switch embeddingType
        case 1
            c = 10000; % Empirically determined
        case 2
            c = 1; % Empirically determined
        case 3
            c = 1; % Empirically determined
        case 4
            %c = 1; % Empirically determined
            c = 10000; % Empirically determined
        case 5
            c = 1; % Empirically determined
    end
    noFolds = 5;

    % Cross validation with C sweeping
    perfCross = [];
    perfTest = [];
    for c = 10 .^ (-6:6);
    %for c = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
        [hybridModelTest, hybridModelCrossval, hybridAccCrossval, hybridRandomCrossval] = ...
                    perclass(valLabel * 2 - 1, valHybridFeatures, c, noFolds, verboseSVM);
        hybridPerfCrossval = mean(hybridAccCrossval);
        fprintf(fileId, 'Val (visual+textual) %f : %f\n', c, mean(hybridPerfCrossval));

        perfCross = [perfCross, mean(hybridPerfCrossval)];
    %end
    %perfCross
    %return
    % Cross validation
    %[hybridModelTest, hybridModelCrossval, hybridAccCrossval, hybridRandomCrossval] = ...
    %            perclass(valLabel * 2 - 1, valHybridFeatures, c, noFolds, verboseSVM);
    %hybridPerfCrossval = mean(hybridAccCrossval);

    % Testing
    [~, ~, hybridScoreTest] = predict(testLabel * 2 - 1, sparse(testHybridFeatures), hybridModelTest{1});
    [hybridPerfTest, baselineTest] = precision(hybridScoreTest, testLabel * 2 - 1);
    fprintf(fileId, 'Test (visual+textual) : %f\n', mean(hybridPerfTest));
    perfTest = [perfTest, mean(hybridPerfTest)];

    corr(hybridScoreTest, testScore, 'type', 'Spearman');
    corr(hybridScoreTest, testScore, 'type', 'Kendall');
    end
    perfTest
    perfCross
end
