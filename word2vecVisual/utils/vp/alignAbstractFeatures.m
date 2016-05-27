% Script to align the textual and visual features in the Abstract image dataset
% (ADS)
function alignAbstractFeatures(vpPath, featPath, savePath)
    idMapFile = 'SceneMapV1_10020.txt';
    nameMapFile = 'SceneMap.txt';

    idMapFile = fopen(idMapFile, 'rb');
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
    fprintf('Reading VP dataset...\n');
    gtPath = fullfile(vpPath, 'data/visual_paraphrasing/split.mat');
    dataPath = fullfile(vpPath, 'data/visual_paraphrasing/dataset.mat');
    load(gtPath);
    dataset = load(dataPath);

    % Get the indices of gt intersect of traininds
    gtInds = find(dataset.gt == 1);
    gtTrainInds = intersect(gtInds, trainind);
    % Find corresponding ids in gtInds
    [~, gtTrainInds] = ismember(gtTrainInds, gtInds);

    % Read the feature file and save only useful training indices
    fullFeatPath = fullfile(featPath, 'abstract_features.txt');
    fullSavePath = fullfile(savePath, 'abstract_features_train.txt');

    features = dlmread(fullFeatPath, ' ', 1, 0);
    %fprintf(saveFile, '%d\n', size(features, 2));
    %fclose(saveFile);
    % The last row is additionally read by dlmread due to ' ' delimiter
    trainFeatures = features(trainMapIds(gtTrainInds), 1:end-1);

    % Write the new features back
    %dlmwrite(savePath, trainFeatures, 'delimiter', ' ', '-append');

    % Save the features to txt file
    noTrain = size(trainFeatures, 1);
    noDims = size(trainFeatures, 2);
    saveId = fopen(fullSavePath, 'wb');
    % save the features twice
    saveFeatures(saveId, [trainFeatures; trainFeatures]);

    fclose(saveId);
    fclose(idMapFile);
    fclose(nameMapFile);
end
