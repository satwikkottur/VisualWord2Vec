% Script to read the matlab saved file and write the train and test features

% Read the mat file
inPath = 'cca_workspace_wiki_after.mat';
load(inPath);

% Get the features for test captions
textTestCCA = bsxfun(@minus, testCaptions, mean(trainCaptions)) * capTrans;

% Save these features to a file
dataPath = '/home/satwik/VisualWord2Vec/data/coco-cca/inter/';
dlmwrite(fullfile(dataPath, 'test_captions_wiki_after_cca.txt'), textTestCCA, ...
                            'delimiter', ' ', 'precision', '%.6f');
dlmwrite(fullfile(dataPath, 'train_captions_wiki_after_cca.txt'), textTrainCCA, ...
                            'delimiter', ' ', 'precision', '%.6f');
