% Script to read the embedded files from C word2vec code
%function readEmbedFile(fileName)
    fileName = '/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/word2vec_pre.txt';
    % Opening the file
    filePt = fopen(fileName);

    % Creating a map
    embedMap = containers.Map();

    % reading the file
    %while(1)
        % read the string
    %    word = fgetl(filePt);
    %    fprintf('%s\n', word);
        %fscanf(filePt, '%s\n', word);
   %     if(isempty(word))
   %         break;
   %     end

        % read and extract the feature
   %     feature = fgetl(filePt);
   %     feature = strsplit(feature, ' ');
   %     feature = str2double(feature(1:end-1));

        % Printing for debugging
        %fprintf('%s\n', word);
   % end

    % reading the file at once
    features = dlmread(fileName);


    fclose(filePt);
%end
