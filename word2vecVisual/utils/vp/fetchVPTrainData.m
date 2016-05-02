% Script to extract all the relevant data to perform VP training
function fetchVPTrainData(absPath, vpPath, savePath)
% Input:
%   absPath - path to abstract scene dataset
%   vpPath - path to the VP dataset
%   savePath - path to save all the files

    % Extract all the features
    extractAbstractFeatures(absPath, vpPath, savePath);
    % Consider only train features
    alignAbstractFeatures(vpPath, savePath, savePath);
    % Get the training sentences for learning vis-w2v
    saveVPTrainSentences(vpPath, savePath);
end
