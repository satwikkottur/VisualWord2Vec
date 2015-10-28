% This script saves the validation and test datasets to be used for C code

rootPath = '/home/satwik/VisualWord2Vec';
valPath = fullfile(rootPath, 'original/val.mat');
testPath = fullfile(rootPath, 'original/test.mat');

val = load(valPath);
valSavePath = fullfile(rootPath, 'data/val_features.txt');
valFile = fopen(valSavePath, 'wb');
% Save validation dataset
for i = 1:length(val.data)
    fprintf(valFile, '<%s:%s:%s> %d\n', strtrim(val.data{i, 1}(1, :)), ...
                                   strtrim(val.data{i, 1}(2, :)), ...
                                   strtrim(val.data{i, 1}(3, :)), ...
                                   val.data{i, 3} > 0);
end
fclose(valFile);

test = load(testPath);
testSavePath = fullfile(rootPath, 'data/test_features.txt');
testFile = fopen(testSavePath, 'wb');
% Save test dataset
for i = 1:length(test.data)
    fprintf(testFile, '<%s:%s:%s> %d\n', strtrim(test.data{i, 1}(1, :)), ...
                                   strtrim(test.data{i, 1}(2, :)), ...
                                   strtrim(test.data{i, 1}(3, :)), ...
                                   test.data{i, 3} > 0);
end
fclose(testFile);
