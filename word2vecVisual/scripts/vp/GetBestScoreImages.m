% Script to get the best scoring images and sentences
% Script to align the textual and visual features in the Abstract image dataset
% (ADS)

% Read the maps given by Xiao
rootPath = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/';
idMapFile = fullfile(rootPath, 'SceneMapV1_10020.txt');
nameMapFile = fullfile(rootPath, 'SceneMap.txt');

idMapFile= fopen(idMapFile, 'rb');
nameMapFile = fopen(nameMapFile, 'rb');

names = textscan(idMapFile, '%s', 'delimiter', '\n');
names = names{1};
% Create a map
nameIdMap = containers.Map(names, 1:length(names));

train = textscan(nameMapFile, '%s', 'delimiter', '\n');
train = train{1};
% Split to extract the second part
splitTrain = regexp(train, '[\w]*\S*[\w]*\S*[\w]*', 'match');
train = cellfun(@(x) x{2}, splitTrain, 'UniformOutput', false);

% Now get the ids for each of the training example
trainMapIds = cellfun(@(x) nameIdMap(x), train);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Now get the gt indices
gtPath = fullfile(rootPath, '../vp-fitb/data/visual_paraphrasing/split.mat');
dataPath = fullfile(rootPath, '../vp-fitb/data/visual_paraphrasing/dataset.mat');
load(gtPath);
load(dataPath);

% Get the indices of gt intersect of test inds
gtInds = find(gt == 1);
gtTrain = intersect(gtInds, testind);
% Find corresponding ids in gtInds
[~, gtTrainInds] = ismember(gtTrainInds, gtInds);

% Read the feature file and save only useful training indices
featPath = '/home/satwik/VisualWord2Vec/data/abstract_features.txt';
savePath = '/home/satwik/VisualWord2Vec/data/abstract_features_train.txt';

%features = dlmread(featPath, ' ', 1, 0);
%fprintf(saveFile, '%d\n', size(features, 2));
%fclose(saveFile);
%trainFeatures = features(trainMapIds(gtTrainInds), :);

% Write the new features back
%dlmwrite(savePath, trainFeatures, 'delimiter', ' ', '-append');

% Save the features to txt file
% First line is number of features
%noTrain = size(trainFeatures, 1);
%noDims = size(trainFeatures, 2);
%saveId = fopen(savePath, 'wb');
%fprintf(saveId, '%d\n', noDims);

% More efficient
%fclose(saveId);
%dlmwrite(savePath, features, 'delimiter', ' ', '-append');
%for i = 1:noTrain
%    fprintf('Saving features : %d / %d...\n', i, noTrain);
%    for j = 1:noDims-1
%        fprintf(saveId, '%f ', trainFeatures(i, j));
%    end
    % Save the last dimension with newline character
%    fprintf(saveId, '%f\n', trainFeatures(i, noDims));
%end

%fclose(saveId);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if (0)
    % Debugging (read all the sentences from ABD, split out only trainInds and check
    % with Xiao's training data
    absPath = '/home/satwik/AbstractScenes_v1.1/SimpleSentences/SimpleSentences1_10020.txt';
    absFile = fopen(absPath, 'rb');

    dataset = textscan(absFile, '%s', 'delimiter', '\n');
    dataset = dataset{1};

    % Remove empty elements
    toRemove = cellfun(@isempty, dataset);
    dataset(toRemove) = [];

    % Remove the leading / trailing spaces
    dataset = strtrim(dataset);

    % Merge the sentences with same id

    % Split to extract the ids and sentences
    splitData = regexp(dataset, '\d*\d*', 'match');
    %sentences = cellfun(@(x) x{3}, splitData, 'UniformOutput', false);
    %ids = cellfun(@(x)
    fclose(absFile);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fclose(idMapFile);
fclose(nameMapFile);
