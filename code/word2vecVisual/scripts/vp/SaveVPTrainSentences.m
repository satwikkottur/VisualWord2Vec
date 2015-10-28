% Script to save the VP trained sentences for tokenization and lemmatization
% load(fullfile('../dumps', 'trainWord2Vec.mat'));

%filePath = fullfile('../dumps', 'vp_train_sentences_raw.txt');
%fileId = fopen(filePath, 'wb');

%for i = 1:length(sentences_train)
%    fprintf(fileId, '%s\n', sentences_train{i});
%end
%
%fclose(fileId);

% Load the data and split
dataPath = '../../data/visual_paraphrasing/';
load(fullfile(dataPath, 'dataset.mat'));
load(fullfile(dataPath, 'split.mat'));

% Get sentences that are gt = 1 and in the training
trainSents1 = sentences_1(trainind);
trainSents2 = sentences_2(trainind);

trainSents = [trainSents1(gt(trainind) == 1); trainSents2(gt(trainind) == 1)];

% Unroll the sentences onto one line
unroll=@(y)cellfun(@(x)cell2mat(x),y,'UniformOutput',false);
trainSents = unroll(trainSents);

savePath = 'lemmatized_training.txt';
saveId = fopen(savePath, 'wb');

% Writing the sentences to a file
cellfun(@(x) fprintf(saveId, '%s\n', x), trainSents);

fclose(saveId);
