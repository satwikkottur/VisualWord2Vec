function[topRecall, medRecall] = testCCA(trainCCAFeats, testCCAFeats, testGtruth)
% Function that takes the CCA space test captions and train captions
% and computes the nearest caption to retrieve image for given query
% 
% Usage:
% [topRecall, medRecall] = testCCA(trainCCAFeats, testCCAFeats, testGtruth);
% 
% Input:
    % trainCCAFeats, testCCAFeats - training and training caption features in CCA space
    % testGTruth - ground truth for the testing images
% 
% Output:
    % topRecall - top 1, 5, 10, 50 recalls
    % medRecall - median recall

    cosDist = 1 - pdist2(testCCAFeats, trainCCAFeats, 'cosine');
    [~, ranks] = sort(cosDist, 2, 'descend');

    % Get the ground truth ranks
    gRank = arrayfun(@(ind) find(ranks(ind, :) == testGtruth(ind)), 1:length(testGtruth));

    topRecall(1) = sum(gRank <= 1);
    topRecall(2) = sum(gRank <= 5);
    topRecall(3) = sum(gRank <= 10);
    medRecall = median(gRank);
end
