% Script that reads the dumps from C code and does qualitative analysis 
% for common sense task
% We find triplets A, B, C such that
% d(A, B) > d(B, C) before refining and
% d(A, B) < d(B, C) after refining

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
%noExp = length(sortInd);
noExp = 100;
featDim = 200;

% Collect unique embeddings for top sorted tuples
% Consider P,R,S separately
% Collect the embeddings for top sorted tuples
pLabels = {};
rLabels = {};
sLabels = {};
uniqPreP = zeros(noExp, featDim);
uniqPreR = zeros(noExp, featDim);
uniqPreS = zeros(noExp, featDim);
uniqPostP = zeros(noExp, featDim);
uniqPostR = zeros(noExp, featDim);
uniqPostS = zeros(noExp, featDim);

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
    if(~ismember(p, pLabels))
        pLabels = [pLabels, {p}];
        uniqPreP(length(pLabels), :) = preEmbed.P(p);
        uniqPostP(length(pLabels), :) = postEmbed.P(p);
    end

    % R
    if(~ismember(r, rLabels))
        rLabels = [rLabels, {r}];
        uniqPreR(length(rLabels), :) = preEmbed.R(r);
        uniqPostR(length(rLabels), :) = postEmbed.R(r);
    end

    % S
    if(~ismember(s, sLabels))
        sLabels = [sLabels, {s}];
        uniqPreS(length(sLabels), :) = preEmbed.S(s);
        uniqPostS(length(sLabels), :) = postEmbed.S(s);
    end
end

% Trim the matrices
uniqPreP = uniqPreP(1:length(pLabels), :);
uniqPreR = uniqPreR(1:length(rLabels), :);
uniqPreS = uniqPreS(1:length(sLabels), :);

uniqPostP = uniqPostP(1:length(pLabels), :);
uniqPostR = uniqPostR(1:length(rLabels), :);
uniqPostS = uniqPostS(1:length(sLabels), :);

% Embed through t-sne (either individual / together)
individual = false;
if(individual)
    noDims = 2;
    noInitDims = [];
    perplexity = 50;

    tsnePreP = tsne(uniqPreP, [], noDims, noInitDims, perplexity);
    tsnePostP = tsne(uniqPostP, [], noDims, noInitDims, perplexity);

    tsnePreS = tsne(uniqPreS, [], noDims, noInitDims, perplexity);
    tsnePostS = tsne(uniqPostS, [], noDims, noInitDims, perplexity);

    tsnePreR = tsne(uniqPreR, [], noDims, noInitDims, perplexity);
    tsnePostR = tsne(uniqPostR, [], noDims, noInitDims, perplexity);

    % Displacement for points
    dx = 0.1; dy = 0.1;
    figure(1); scatter(tsnePreP(:, 1), tsnePreP(:, 2))
    text(tsnePreP(:, 1) + dx, tsnePreP(:, 2) + dy, pLabels)
    figure(2); scatter(tsnePostP(:, 1), tsnePostP(:, 2))
    text(tsnePostP(:, 1) + dx, tsnePostP(:, 2) + dy, pLabels)

    figure(3); scatter(tsnePreS(:, 1), tsnePreS(:, 2))
    text(tsnePreS(:, 1) + dx, tsnePreS(:, 2) + dy, sLabels)
    figure(4); scatter(tsnePostS(:, 1), tsnePostS(:, 2))
    text(tsnePostS(:, 1) + dx, tsnePostS(:, 2) + dy, sLabels)

    figure(5); scatter(tsnePreR(:, 1), tsnePreR(:, 2))
    text(tsnePreR(:, 1) + dx, tsnePreR(:, 2) + dy, rLabels)
    figure(6); scatter(tsnePostR(:, 1), tsnePostR(:, 2))
    text(tsnePostR(:, 1) + dx, tsnePostR(:, 2) + dy, rLabels)
else
    noDims = 2;
    noInitDims = 50;
    perplexity = 5;

    tsnePre = tsne([uniqPreP; uniqPreS; uniqPreR], [], noDims, noInitDims, perplexity);
    tsnePost = tsne([uniqPostP; uniqPostS; uniqPostR], [], noDims, noInitDims, perplexity);

    labels = [pLabels, sLabels, rLabels];
    groups = [ones(length(pLabels), 1); ...
            2 * ones(length(sLabels), 1); ...
            3 * ones(length(rLabels), 1)];

    % Displacement for points
    dx = 0.1; dy = 0.1;
    figure(7); gscatter(tsnePre(:, 1), tsnePre(:, 2), groups)
    text(tsnePre(:, 1) + dx, tsnePre(:, 2) + dy, labels)
    figure(8); gscatter(tsnePost(:, 1), tsnePost(:, 2), groups)
    text(tsnePost(:, 1) + dx, tsnePost(:, 2) + dy, labels)
end
