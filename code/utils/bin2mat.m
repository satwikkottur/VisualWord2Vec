function bin2mat(binPath, savePath)
    % This function converts the saved word2vec bin files to mat files
    %rootPath = '/home/satwik/VisualWord2Vec/word2vecVisual/modelsNdata/';
    %binPath = fullfile(rootPath, 'word2vec_coco_after_300.bin');
    %savePath = fullfile(rootPath, 'word2vec_coco_after_300.mat');

    % Open the file
    fileId = fopen(binPath);
    embeds = containers.Map();

    tline = fgetl(fileId);
    split = strsplit(tline, ' ');
    numWords = str2num(char(split(1))); 

    for i = 1:numWords
        % Read the line and save the vector
        tline = fgetl(fileId);

        % Extract the string and features
        splits = strsplit(tline, ' ');
        string = char(splits(1));
        feature = cellfun(@str2num, splits(2:end-1));

        embeds(string) = feature;

        % Print the current size
        if rem(embeds.Count, 100) == 0
            fprintf('Current feature: %d / %d\n', embeds.Count, numWords);
        end
    end

    % Close the file
    fclose(fileId);

    % Save matfile
    save(savePath, 'embeds');
    fprintf('Saved the mat file at %s\n', savePath);
end
