% Script that demonstrates the use of textual features for common sense
% Used to compare before and after refining the network

%******************************************************
rootPath = '/home/satwik/VisualWord2Vec';
dataPath = '/home/satwik/VisualWord2Vec/data';

w2vModelPath = fullfile(rootPath, 'code/word2vecVisual/modelsNdata', 'coco_w2v_pre.mat');
psrFeaturePath = fullfile(dataPath, 'PSR_features.txt');
numFeaturePath = fullfile(dataPath, 'Num_features.txt');
resultPath = fullfile(rootPath, 'code/word2vecVisual/modelsNdata', 'word2vec_output.bin');
%******************************************************
convertWord2VecResults(resultPath, w2vModelPath);
w2vModel = load(w2vModelPath);
fprintf('Loaded word2vec model succesfully!\n');
% Reading the labels
[Plabel, Slabel, Rlabel, Rfeatures] = readFromFile(psrFeaturePath, numFeaturePath);

% Encoding P, R, S labels using one hot encoding(unique labels)
[Pencoding, Pdict, Pinds] = oneHotEncode(Plabel);
[Sencoding, Sdict, Sinds] = oneHotEncode(Slabel);
[Rencoding, Rdict, Rinds] = oneHotEncode(Rlabel);

% Vector embedding for P,R,S
Pembed = embedLabels(Pdict, w2vModel);
Sembed = embedLabels(Sdict, w2vModel);
Rembed = embedLabels(Rdict, w2vModel);

% Load validation and test data
val = load('/home/satwik/VisualWord2Vec/original/val.mat');
test = load('/home/satwik/VisualWord2Vec/original/test.mat');

noTest = size(test.data, 1);
noVal = size(val.data, 1);

% Cleaning strings
[valP, valR, valS] = cleanStrings(val.data);
[testP, testR, testS] = cleanStrings(test.data);

fprintf('Cleaned the strings...\n');

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

% Computing human scores for the labels
valScore = cell2mat(val.data(:, 3));
valLabel = valScore > 0;

testScore = cell2mat(test.data(:, 3)) > 0;
testLabel = testScore > 0 ;

fprintf('Evaluating the text scores...\n');
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
precValues = [];
% Faster implementation
for threshold = -2:0.1:1
    textValScore = mean(max(...
                            valRscoreText(:, valRlabels) + ...
                            valPscoreText(:, valPlabels) + ...
                            valSscoreText(:, valSlabels) - ...
                            threshold, 0), 1)';
    [precVal, baseVal] = precision(textValScore, valLabel);
    fprintf('Test (textual) %f : %f\n', threshold, mean(precVal(:)));

    precValues = [precValues, mean(precVal(:))];

    % Use the same threshold for test set
    % Faster implementation
    textTestScore = mean(max(...
                        testRscoreText(:, testRlabels) + ...
                        testPscoreText(:, testPlabels) + ...
                        testSscoreText(:, testSlabels) - ...
                        threshold, 0), 1)';
    [precTest, baseTest] = precision(textTestScore, testLabel);
    fprintf('Test (textual) : %f\n', mean(precTest(:)));
end
