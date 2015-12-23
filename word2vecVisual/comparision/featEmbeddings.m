% Script to dump the embeddings for the features
% Load the word2vec mat file`

rootPath = '/home/satwik/VisualWord2Vec/';
w2vPath = fullfile(rootPath, 'original/coco_w2v.mat')
load(w2vPath);

% Read the feature vocab file
vocabPath = 'word2vec_vocab_0_0_1_25.txt';
vocabFile = fopen(vocabPath, 'rb');
featVocab = textscan(vocabFile, '%s', 'delimiter', '\n');
fclose(vocabFile);
featVocab = featVocab{1};

% Open the feature dumping file
dumpPath = 'featEmbed_matlab.txt';

matlab = dlmread(dumpPath);
c = dlmread('word2vec_pre_0_0_1_25.txt');

% Print if the difference is great
diff = sum(abs(matlab - c), 2) > 1e-3;
% Take only 1 out of every 3
diff = diff(1:3:end);
featVocab(diff)

% Consider the first element 
noDims = size(fv, 2);

% Writing the file for the current set of feature words
if(0)
    dumpFile = fopen(dumpPath, 'wb');
    for i = 1:length(featVocab)
        AllEmbed = zeros(3 * length(featVocab), noDims);
        fprintf('Current : %d / %d\n', i, length(featVocab));
        embed = embed_str(featVocab{i}, tokens, fv);

        %AllEmbed((i-1)*3 + (1:3), :) = repmat(embed, [3, 1]);
        for j = 1:noDims-1
            fprintf(dumpFile, '%7.6f ', embed(j));
        end
        fprintf(dumpFile, '%7.6f\n', embed(noDims));
        for j = 1:noDims-1
            fprintf(dumpFile, '%7.6f ', embed(j));
        end
        fprintf(dumpFile, '%7.6f\n', embed(noDims));
        for j = 1:noDims-1
            fprintf(dumpFile, '%7.6f ', embed(j));
        end
        fprintf(dumpFile, '%7.6f\n', embed(noDims));
    end
    fclose(dumpFile);
end

%dlmwrite(dumpPath, AllEmbed, 'delimiter', ' ');
