function [sortedTuples, sortInd] = sortImprovedTuples(tuples)
    % Function to sort the improved tuples based on their score
    
    % Sort based on the improvement
    [~, sortInd] = sort(tuples{5}-tuples{4}, 'descend');

    sortedTuples = {tuples{1}(sortInd), tuples{2}(sortInd), tuples{3}(sortInd), ...
                                tuples{4}(sortInd), tuples{5}(sortInd)};
end
