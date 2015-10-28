% Script that reads word2vec results from the c code result file into matlab
% Changed to a function
function convertWord2VecResults(resultPath, savePath)
    % Reading the file
    resultPath
    d = textread(resultPath, '%s');

    % then this should get the words (tokens_word2vec) and a corresponding matrix of vectors (vectors_word2vec)
    t = reshape(d(3:end), [str2num(d{2})+1 str2num(d{1})]);
    tokens = t(1,:);
    fv = cell2mat(cellfun(@str2num,t(2:end,:), 'UniformOutput', false))';

    % Saving the dictionary and features
    save(savePath, 'fv', 'tokens');
end
