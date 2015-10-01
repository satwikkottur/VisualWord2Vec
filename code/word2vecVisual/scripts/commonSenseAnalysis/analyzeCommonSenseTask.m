% Script that reads the dumps from C code and does qualitative analysis 
% for common sense task

% Setting up paths
rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual';
modelPath = fullfile(rootPath, 'modelsNdata');
addpath(genpath('/home/satwik/VisualWord2Vec/libs'));

preFile = fullfile(modelPath, 'word2vec_pre_0_0_1_25.txt');
postFile = fullfile(modelPath, 'word2vec_post_0_0_1_25.txt');
vocabFile = fullfile(modelPath, 'word2vec_vocab_0_0_1_25.txt');
tupleFile = fullfile(modelPath, 'improved_test_tuples.txt');
embedFile = fullfile(modelPath, 'improved_test_embed.txt');

verbose = false;

if(~exist('preEmbed', 'var'))
    % Read the embeddings before / after refining along with vocab
    [preEmbed, postEmbed, featureWords] = ...
                        readEmbedFile(preFile, postFile, vocabFile, true);

    % Reading the tuples that caused an increase
    tuples = readImprovedTuples(tupleFile, embedFile);
end

% Sort the improved tuples in order and pick max words
[sortedTuples, sortInd] = sortImprovedTuples(tuples);

% Print first N tuples
noExp = 30;
featDim = 200;
% Collect the embeddings for top sorted tuples
preCollect = zeros(3 * noExp, featDim);
postCollect = zeros(3 * noExp, featDim);
labels = cell(3 * noExp, 1);
for i = 1:noExp
    if verbose
        fprintf('<%s , %s , %s> : %f %f\n', char(sortedTuples{1}(i)), ...
                                            char(sortedTuples{2}(i)), ...
                                            char(sortedTuples{3}(i)), ...
                                            sortedTuples{4}(i), ...
                                            sortedTuples{5}(i));
    end

    preCollect(3*(i-1) + (1:3), :) = [preEmbed.P(char(sortedTuples{1}(i))); ...
                                        preEmbed.R(char(sortedTuples{2}(i))); ...
                                        preEmbed.S(char(sortedTuples{3}(i)))];

    postCollect(3*(i-1) + (1:3), :) = [postEmbed.P(char(sortedTuples{1}(i))); ...
                                        postEmbed.R(char(sortedTuples{2}(i))); ...
                                        postEmbed.S(char(sortedTuples{3}(i)))];
    labels(3 *(i-1) + (1:3)) = {char(sortedTuples{1}(i)); ...
                                        char(sortedTuples{2}(i)); ...
                                        char(sortedTuples{3}(i))};
end

% Embed through t-sne
noDims = 2;
noInitDims = 20;
perplexity = 50;
tsnePre = tsne(preCollect, [], noDims, noInitDims, perplexity);
tsnePost = tsne(postCollect, [], noDims, noInitDims, perplexity);

%figure(1); scatter(tsnePre(:, 1), tsnePre(:, 2))
%figure(2); scatter(tsnePost(:, 1), tsnePost(:, 2))
figure(1); text(tsnePre(:, 1), tsnePre(:, 2), labels)
figure(2); text(tsnePost(:, 1), tsnePost(:, 2), labels)
% Read
