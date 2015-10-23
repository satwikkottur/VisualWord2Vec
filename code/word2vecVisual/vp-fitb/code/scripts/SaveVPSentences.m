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

% Unroll the sentences onto one line
unroll=@(y)cellfun(@(x)cell2mat(x),y,'UniformOutput',false);
%****************************************
sentences1 = unroll(sentences_1);

savePath = '../dumps/vp_sentences_1.txt';
saveId = fopen(savePath, 'wb');

% Writing the sentences to a file
cellfun(@(x) fprintf(saveId, '%s\n', x), sentences1);

fclose(saveId);
%****************************************
sentences2 = unroll(sentences_2);

savePath = '../dumps/vp_sentences_2.txt';
saveId = fopen(savePath, 'wb');

% Writing the sentences to a file
cellfun(@(x) fprintf(saveId, '%s\n', x), sentences2);

fclose(saveId);
%****************************************
