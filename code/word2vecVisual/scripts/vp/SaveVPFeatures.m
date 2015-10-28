% Script to save the features for vp task
% All other features, except word2vec for all sentences
function SaveVPFeatures()
    dumpPath = '../dumps/text_other_features_workspace.mat';
    a = load(dumpPath);

    % Save features for sentences1
    % feat_vp_text_coc_1
    savePath = '../dumps/vp_features_coc_1.txt';
    %SaveFeatures(savePath, a.feat_vp_text_coc_1);

    % feat_vp_text_tf_1
    savePath = '../dumps/vp_features_tf_1.txt';
    SaveFeatures(savePath, full(a.feat_vp_text_tf_1));

    % Save features for sentences2
    % feat_vp_text_coc_2
    savePath = '../dumps/vp_features_coc_2.txt';
    SaveFeatures(savePath, a.feat_vp_text_coc_2);

    % feat_vp_text_tf_2
    savePath = '../dumps/vp_features_tf_2.txt';
    SaveFeatures(savePath, full(a.feat_vp_text_tf_2));
end

% Saving the features
function SaveFeatures(savePath, features)
    % Opening the file
    saveId = fopen(savePath, 'wb');
    
    % Writing the dimension
    [noInsts, noDims] = size(features);
    fprintf(saveId, '%d\n', noDims);

    for i = 1:noInsts
        if(rem(i, 100) == 0)
            fprintf('%s : %d / %d\n', savePath, i, noInsts);
        end
        for j = 1:noDims-1
            fprintf(saveId, '%f ', features(i, j));
        end
        fprintf(saveId, '%f\n', features(i, noDims));
    end

    % Close the file
    fclose(saveId);
end
