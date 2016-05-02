% Script to get the features from abstract dataset, concatenate all the feature files
function extractAbstractFeatures(dataPath, savePath)
    % Get the listing of the feature files
    listing = dir(fullfile(dataPath, 'VisualFeatures', '*.txt'));
    listing = {listing.name}';

    % Remove Rel GMM as flip account version is used
    features = cell(length(listing), 1);
    noFeatFiles = 0;

    fprintf('Reading features...\n');
    for i = 1:length(listing)
        fileName = listing{i};
        if(isempty(strfind(fileName, 'names')) && isempty(strfind(fileName, '1508')))
            fprintf('\t%s\n', fileName);
            noFeatFiles = noFeatFiles + 1;

            % Open file, read and store the results
            features{noFeatFiles} = dlmread(fullfile(dataPath, 'VisualFeatures', fileName));
        end
    end

    % Trimming cell
    features = horzcat(features{1:noFeatFiles});
    noTrain = size(features, 1);
    noDims = size(features, 2);

    % Save the features to txt file
    % First line is number of features
    saveId = fopen(fullfile(savePath, 'abstract_features.txt'), 'wb');
    saveFeatures(saveId, features);
    fclose(saveId);
end
