% Script to save the VP trained sentences for tokenization and lemmatization
function saveVPTrainSentences(vpPath, savePath)
    % Load the data and split
    data = load(fullfile(vpPath, 'data/visual_paraphrasing', 'dataset.mat'));
    load(fullfile(vpPath, 'data/visual_paraphrasing', 'split.mat'));

    % Get sentences that are gt = 1 and in the training
    trainSents1 = data.sentences_1(trainind);
    trainSents2 = data.sentences_2(trainind);

    trainSents = [trainSents1(data.gt(trainind) == 1);...
                    trainSents2(data.gt(trainind) == 1)];

    % Unroll the sentences onto one line
    unroll=@(y)cellfun(@(x)cell2mat(x),y,'UniformOutput',false);
    trainSents = unroll(trainSents);

    saveFile = 'vp_train_sentences_raw.txt';
    saveId = fopen(fullfile(savePath, saveFile), 'wb');

    % Writing the sentences to a file
    cellfun(@(x) fprintf(saveId, '%s\n', x), trainSents);

    fclose(saveId);
end
