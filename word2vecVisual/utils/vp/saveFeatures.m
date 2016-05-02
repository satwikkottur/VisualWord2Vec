% Script to save a feature matrix to a file given pointer
function saveFeatures(fileId, featMat)
    % First line is number of features
    fprintf(fileId, '%d\n', size(featMat, 2));

    reverseStr = '';
    for i = 1:size(featMat, 1)
        % Display the progress
        if rem(i, 10) == 0
            percentDone = 100 * i / size(featMat, 1);
            msg = sprintf('Saving features : %3.1f percent done...', percentDone); %Don't forget this semicolon
            fprintf([reverseStr, msg]);
            reverseStr = repmat(sprintf('\b'), 1, length(msg));
        end

        fprintf(fileId, '%f ', featMat(i, :));
        fprintf(fileId, '\n');
    end
    fprintf('\n');
end
