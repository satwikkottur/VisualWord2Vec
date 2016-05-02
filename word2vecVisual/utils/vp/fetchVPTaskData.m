% Script to extract all the relevant data to perform VP task
function fetchVPTaskData(vpPath, matPath, savePath)
% Input:
%   vpPath - path to the VP dataset
%   matPath - path to matfile 'vp_txt_features_mat'
%   savePath - path to save all the files

    %--------------- Saving the ground truth ---------------
    fprintf('Reading the VP dataset..');
    load(fullfile(vpPath, 'data/visual_paraphrasing', 'dataset.mat'));
    load(fullfile(vpPath, 'data/visual_paraphrasing', 'split.mat'));

    fprintf('Saving the ground truth...');
    saveId = fopen(fullfile(savePath, 'vp_ground_truth.txt'), 'wb');
    for i = 1:length(gt)
        fprintf(saveId, '%d\n', gt(i));
    end
    fclose(saveId);

    %--------------- Saving the sentences ---------------
    % Unroll the sentences onto one line
    fprintf('Saving the sentences...');
    unroll=@(y)cellfun(@(x)cell2mat(x),y,'UniformOutput',false);
    
    sentences1 = unroll(sentences_1);

    saveId = fopen(fullfile(savePath, 'vp_sentences_1.txt'), 'wb');
    % Writing the sentences to a file
    cellfun(@(x) fprintf(saveId, '%s\n', x), sentences1);
    fclose(saveId);

    sentences2 = unroll(sentences_2);

    saveId = fopen(fullfile(savePath, 'vp_sentences_2.txt'), 'wb');
    % Writing the sentences to a file
    cellfun(@(x) fprintf(saveId, '%s\n', x), sentences2);
    fclose(saveId);
    %--------------- Saving the train / test split ---------------
    fprintf('Saving the train / test split...');

    totalInst = length(testind) + length(trainind);
    isTrain = zeros(totalInst, 1);
    isTrain(trainind) = 1;

    saveId = fopen(fullfile(savePath, 'vp_split.txt'), 'wb');
    for i = 1:totalInst
        fprintf(saveId, '%d\n', isTrain(i));
    end
    fclose(saveId);
    %--------------- Saving the validation set ---------------
    % Get the ratio for the training inds
    %gtTrain = gt(trainind);
    %sum(gtTrain)./length(gtTrain)
    % 1 from gt = 1
    % 2 from gt = 0
    % Only from traininds
    gtInds = find(gt == 1);
    gtTrainInds = intersect(gtInds, trainind);

    falseInds = find(gt == 0);
    falseTrainInds = intersect(falseInds, trainind);

    % Choose 1k from gtTrainInds and 2k from falseTrainInds
    noVals = 1000;
    gtValInds = datasample(gtTrainInds, floor(noVals/3), 'Replace', false);
    falseValInds = datasample(falseTrainInds, 2* floor(noVals/3), 'Replace', false);

    valInds = union(gtValInds, falseValInds);

    valSet = zeros(length(gt), 1);
    valSet(valInds) = 1;

    saveId = fopen(fullfile(savePath, 'vp_val_inds_1k.txt'), 'wb');
    for i = 1:length(gt)
        fprintf(saveId, '%d\n', valSet(i));
    end
    fclose(saveId);
    %--------------- Saving the validation set ---------------
    otherFeats = load(fullfile(matPath, 'vp_txt_features.mat');

    % Save features for sentences1
    % feat_vp_text_coc_1
    saveId = fopen(fullfile(savePath, 'vp_features_coc_1.txt'), 'w');
    SaveFeatures(savePath, otherFeats.feat_vp_text_coc_1);
    fclose(saveId);

    % Save features for sentences2
    % feat_vp_text_coc_2
    saveId = fopen(fullfile(savePath, 'vp_features_coc_2.txt'), 'w');
    SaveFeatures(savePath, otherFeats.feat_vp_text_coc_2);
    fclose(saveId);

    % Save features for sentences1
    % feat_vp_text_tf_1
    saveId = fopen(fullfile(savePath, 'vp_features_tf_1.txt'), 'w');
    SaveFeatures(savePath, otherFeats.feat_vp_text_tf_1);
    fclose(saveId);

    % Save features for sentences2
    % feat_vp_text_tf_2
    saveId = fopen(fullfile(savePath, 'vp_features_tf_2.txt'), 'w');
    SaveFeatures(savePath, otherFeats.feat_vp_text_tf_2);
    fclose(saveId);
end
