% Script to get the dictionary for word2vec results written in C
function getWord2VecDict(loadPath, savePath)
    % Reading the file
    d = textread(loadPath, '%s');

    % then this should get the words (tokens_word2vec) and a corresponding matrix of vectors (vectors_word2vec)
    t = reshape(d(3:end), [str2num(d{2})+1 str2num(d{1})]);
    tokens = t(1,:);
    fv = cellfun(@str2num,t(2:end,:), 'UniformOutput', false);

    % Create a container map
    embeds = containers.Map(tokens, fv);

    % Saving the dictionary and features
    save(savePath, 'embeds');
end
