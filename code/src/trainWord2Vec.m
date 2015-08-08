% Script to train word2vec for a given text file

% to basically you start with a corpus let's say word2vec_doc.txt, then you run word2vec (output as a text file)
% compute word2vec
rootPath = '/home/satwik/VisualWord2Vec';
fileName = 'coco_train_minus_cs_test.txt';
%fileName = 'coco_train_minus_cs_test_tokenized_stops.txt';
modelName = strrep(fileName, '.txt', '.bin');
%modelName = 'word2vecCOCO.bin';

srcPath = fullfile(rootPath, 'libs/word2vec', 'word2vec');
filePath = fullfile(rootPath, 'data', fileName);
modelPath = fullfile(rootPath, 'models', modelName);

system([srcPath ...
        ' -train ' filePath ...
        ' -output ' modelPath ...
        ' -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0']);

% then read the result file
% get vectors
%d = textread(modelPath, '%s');

% then this should get the words (tokens_word2vec) and a corresponding matrix of vectors (vectors_word2vec)
%t = reshape(d(3:end), [str2num(d{2})+1 str2num(d{1})]);
%tokens = t(1,:);
%fv = cell2mat(cellfun(@str2num,t(2:end,:), 'UniformOutput', false))';

%save(fullfile(rootPath, 'models', 'coco_w2v_raw.mat'), 'fv', 'tokens');
