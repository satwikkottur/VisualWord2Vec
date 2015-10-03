% Script to check how well matlab splits

% Read the vocab file
featFile = fopen('vocab.txt', 'rb');
splitFile = fopen('matlab_splitting.txt', 'wb');

featVocab = textscan(featFile, '%s', 'Delimiter', '\n');
featVocab = featVocab{1};

for i = 1:length(featVocab)
    % Split
    embed_str(featVocab{i}, splitFile);
    fprintf('%d %d\n', i, length(featVocab));
    % Print the splits
end

%close the files
fclose(featFile);
fclose(splitFile);
