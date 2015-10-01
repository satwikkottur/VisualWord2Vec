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
noExp = 100;
featDim = 200;

% Collect unique embeddings for top sorted tuples
% Consider P,R,S separately
% Collect the embeddings for top sorted tuples
p
uniqPreP = containers.Map(); uniqPostP = containers.Map();
uniqPreR = containers.Map(); uniqPostR = containers.Map();
uniqPreS = containers.Map(); uniqPostS = containers.Map();

for i = 1:noExp
    if verbose
        fprintf('<%s , %s , %s> : %f %f\n', char(sortedTuples{1}(i)), ...
                                            char(sortedTuples{2}(i)), ...
                                            char(sortedTuples{3}(i)), ...
                                            sortedTuples{4}(i), ...
                                            sortedTuples{5}(i));
    end

    p = char(sortedTuples{1}(i));
    r = char(sortedTuples{2}(i));
    s = char(sortedTuples{3}(i));

    % P
    % New label, add it
    if(~uniqPreP.isKey(p))
        uniqPreP(p) = preEmbed.P(p);
        uniqPostP(p) = postEmbed.P(p);
    end

    % R
    if(~uniqPreR.isKey(r))
        uniqPreR(r) = preEmbed.R(r);
        uniqPostR(r) = postEmbed.R(r);
    end

    % S
    if(~uniqPreS.isKey(s))
        uniqPreS(s) = preEmbed.S(s);
        uniqPostS(s) = postEmbed.S(s);
    end
end

% Now collect the embeddings
preCollectP = [];
preCollectS = [];
preCollectR = [];
postCollectP = [];
postCollectS = [];
postCollectR = [];
for i = 1:
% Embed through t-sne
noDims = 2;
noInitDims = 50;
perplexity = 5;
tsnePre = tsne([cell2mat(uniqPreP.values'); ...
                cell2mat(uniqPreS.values'); ...
                cell2mat(uniqPreR.values')],...
                [], noDims, noInitDims, perplexity);

tsnePost = tsne([cell2mat(uniqPostP.values'); ...
                cell2mat(uniqPostS.values'); ...
                cell2mat(uniqPostR.values')],...
                [], noDims, noInitDims, perplexity);

labels = [uniqPreP.keys, uniqPreS.keys, uniqPreR.keys];
groups = [ones(length(uniqPreP.keys), 1); ...
        2 * ones(length(uniqPreS.keys), 1); ...
        3 * ones(length(uniqPreR.keys), 1)];

% Displacement for points
dx = 0.1; dy = 0.1;
figure(1); gscatter(tsnePre(:, 1), tsnePre(:, 2), groups)
text(tsnePre(:, 1) + dx, tsnePre(:, 2) + dy, labels)

figure(2); gscatter(tsnePost(:, 1), tsnePost(:, 2), groups)
text(tsnePost(:, 1) + dx, tsnePost(:, 2) + dy, labels)
% Read
