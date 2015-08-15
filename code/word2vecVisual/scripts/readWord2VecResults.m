% Script that reads word2vec results from the c code result file into matlab

resultFile = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_output_newline.bin';

d = textread(resultFile, '%s');

% then this should get the words (tokens_word2vec) and a corresponding matrix of vectors (vectors_word2vec)
t = reshape(d(3:end), [str2num(d{2})+1 str2num(d{1})]);
tokens = t(1,:);
fv = cell2mat(cellfun(@str2num,t(2:end,:), 'UniformOutput', false))';

save('coco_w2v_newline.mat', 'fv', 'tokens');
