% Script to read the visual features as mat files and writing the into
% text file for cross-language usage

% MAT files taken from /home/satwik/VisualWord2Vec/data/modifiedFeatures/

dataPath = '/home/satwik/VisualWord2Vec/data';
featPath = fullfile(dataPath, 'modifiedFeatures/');

listing = dir(fullfile(featPath, '*.mat'));
featDim = 1222;

% Open a file to write the features
fileId = fopen(fullfile(dataPath, 'float_features.txt'), 'wb');
for i = 1:length(listing)
    fprintf('Current feature : %d / %d...\n', i, length(listing));
    feature = load(fullfile(featPath, listing(i).name));

    % check for the size of the features
    if(length(feature.feat) == featDim)
        % Writing the features to the file
        for n = 1:featDim-1
            fprintf(fileId, '%f ', feature.feat(n));
        end
        fprintf(fileId, '%f\n', feature.feat(featDim));
    end
end

fclose(fileId);
