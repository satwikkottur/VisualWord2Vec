% Script to read the embedded files from C word2vec code
function[preEmbed, postEmbed, featureWords] = readEmbedFile(preFile, postFile, vocabFile, multiModel)
    % Function to read embed files, before and after refining along with the vocabulary

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
end
