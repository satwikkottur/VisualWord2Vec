% Select the training features for abstrast dataset
% Use the extracted abstract features and remove sentences that are empty

% Load the data and split
dataPath = '../vp-fitb/data/visual_paraphrasing/';
load(fullfile(dataPath, 'dataset.mat'));
load(fullfile(dataPath, 'split.mat'));

% Get sentences that are gt = 1 and in the training
trainSents1 = sentences_1(trainind);
trainSents2 = sentences_2(trainind);

trainSents = [trainSents1(gt(trainind) == 1); trainSents2(gt(trainind) == 1)];

% Unroll the sentences onto one line
unroll=@(y)cellfun(@(x)cell2mat(x),y,'UniformOutput',false);
trainSents = unroll(trainSents);

% Get the indices to skip
toRemove = find(cellfun(@isempty, trainSents) == 1);

% Read the total features file
featurePath = '/home/satwik/VisualWord2Vec/data/abstract_features.txt';
features = dlmread(featurePath, ' ', 1, 0);

% Save only required features
features(toRemove, :) = [];
savePath = '/home/satwik/VisualWord2Vec/data/abstract_features_cleaned.txt';

noTrain = size(features, 1);
noDims = size(features, 2);

% Save the features to txt file
% First line is number of features
saveId = fopen(savePath, 'wb');
fprintf(saveId, '%d\n', noDims);

% More efficient
%fclose(saveId);
%dlmwrite(savePath, features, 'delimiter', ' ', '-append');

for i = 1:noTrain
    fprintf('Saving features : %d / %d...\n', i, noTrain);
    for j = 1:noDims-1
        fprintf(saveId, '%f ', features(i, j));
    end
    % Save the last dimension with newline character
    fprintf(saveId, '%f\n', features(i, noDims));
end

fclose(saveId);
