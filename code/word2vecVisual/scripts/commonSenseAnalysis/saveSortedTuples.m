% Function to save sortedTuples
function saveSortedTuples(sortedTuples, savePath)

    %savePath = 'sortedTuples.txt';
    fileId = fopen(savePath, 'wb');

    for i = 1:length(sortedTuples{1})
        fprintf(fileId, '<%s:%s:%s>(%f, %f)\n', sortedTuples{1}{i}, ...
                                            sortedTuples{2}{i}, ... 
                                            sortedTuples{3}{i}, ... 
                                            sortedTuples{4}(i), ... 
                                            sortedTuples{5}(i)); 
    end

    fclose(fileId);
end
