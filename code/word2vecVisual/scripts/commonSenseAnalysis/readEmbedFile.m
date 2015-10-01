% Script to read the embedded files from C word2vec code
function[preEmbed, postEmbed, featureWords] = ...
                        readEmbedFile(preFile, postFile, vocabFile, multiModel)
    % Function to read embed files, before and after refining 
    % along with the vocabulary

    % Multi model flag
    multiModel = true;

    % Reading the contents into the file
    preEmbed = dlmread(preFile);
    postEmbed = dlmread(postFile);

    fileId = fopen(vocabFile);
    featureWords = textscan(fileId, '%s', 'Delimiter', '\n');
    
    % First component has all the words, what we care about
    featureWords = featureWords{1};

    % Close the file
    fclose(fileId);

    % Creating a dictionary for pre embeddings and post embeddings
    preEmbedP = preEmbed(1:3:end, :);
    preEmbedR = preEmbed(2:3:end, :);
    preEmbedS = preEmbed(3:3:end, :);
    postEmbedP = postEmbed(1:3:end, :);
    postEmbedR = postEmbed(2:3:end, :);
    postEmbedS = postEmbed(3:3:end, :);

    preP = containers.Map();
    preR = containers.Map();
    preS = containers.Map();
    postP = containers.Map();
    postR = containers.Map();
    postS = containers.Map();

    for i = 1:size(preEmbedP)
        %fprintf('%d %d\n', i, size(preEmbedP));
        preP(featureWords{i}) = preEmbedP(i, :);
        preR(featureWords{i}) = preEmbedR(i, :);
        preS(featureWords{i}) = preEmbedS(i, :);

        postP(featureWords{i}) = postEmbedP(i, :);
        postR(featureWords{i}) = postEmbedR(i, :);
        postS(featureWords{i}) = postEmbedS(i, :);
    end

    preEmbed = struct('P', preP, 'R', preR, 'S', preS);
    postEmbed = struct('P', postP, 'R', postP, 'S', postS);
end
