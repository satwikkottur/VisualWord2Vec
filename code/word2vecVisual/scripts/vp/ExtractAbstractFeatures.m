% Script to get the features from abstract dataset, concatenate all the feature files

dataPath = '/home/satwik/AbstractScenes_v1.1/VisualFeatures/';
savePath = '/home/satwik/VisualWord2Vec/data/abstract_features.txt';

% Get the listing of the feature files
listing = dir(fullfile(dataPath, '*.txt'));
listing = {listing.name}';

% Remove Rel GMM as flip account version is used
features = cell(length(listing), 1);
noFeatFiles = 0;

for i = 1:length(listing)
    fileName = listing{i};
    if(isempty(strfind(fileName, 'names')) && isempty(strfind(fileName, '1508')))
        noFeatFiles = noFeatFiles + 1;

        % Open file, read and store the results
        features{noFeatFiles} = dlmread(fullfile(dataPath, fileName));
    end
end

% Trimming cell
features = horzcat(features{1:noFeatFiles});
noTrain = size(features, 1);
noDims = size(features, 2);

% Four sentences are empty, find their indices and remove them from the features before sving
load();

toRemove = [];
features(toRemove) = [];

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
