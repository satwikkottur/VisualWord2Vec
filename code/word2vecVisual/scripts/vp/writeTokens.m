% Script to write tokens to file

% Load the mat dump file
addpath('/home/satwik/VisualWord2Vec/code/word2vecVisual/scripts/');
rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/';
matDumpPath = fullfile(rootPath, 'vp-fitb/code/dumps/trainWord2Vec.mat');
load(matDumpPath)

% Open file to dump
savePath = 'token_dump.txt';
saveFile = fopen(savePath, 'wb');

for i = 1:length(tokens)
    fprintf(saveFile, '%s\n', tokens{i});
end

fclose(saveFile);
