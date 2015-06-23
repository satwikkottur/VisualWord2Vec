function [encoding, uniqueLabels, labelIds] = oneHotEncode(labels)
    % Function to perform one hot coding on the labels
    % Find unique labels and assign 1 for input labels at each unique label
    % 
    % Input:
    % labels = List of labels - cell of N x 1
    %
    % Output:
    % encoding = One hot encoding, one for each row ( N x Nunique)
    %           where Nunique is the number of unique labels
    % uniqueLabels = Returns the unique labels from the labels provided
    %           as input
    % labelIds = Returns the labelIds of input labels wrt uniqueLabel dictionary

    uniqueLabels = unique(labels);

    noUniqueLabels = length(uniqueLabels);
    noLabels = length(labels);

    encoding = -1 * ones(noLabels, noUniqueLabels);
    [~, labelIds] = ismember(labels, uniqueLabels);
    
    % Computing the linear indices to turn on the corresponding index,
    % and turning them on
    linearInds = sub2ind(size(encoding), (1:noLabels)', labelIds);
    encoding(linearInds) = 1;
end
