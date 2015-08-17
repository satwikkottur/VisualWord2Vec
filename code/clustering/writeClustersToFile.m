% Script to write the clusters to the file
function writeClustersToFile(clusterIds, fileName)
    % Create the file handler
    fileId = fopen(fileName, 'wb');

    % Write the cluster ids
    for i = 1:size(clusterIds)
        fprintf(fileId, '%d\n', clusterIds(i));
    end
end
