% This function saves the P,R,S and numerical visual features 
% in the specified directories in the format expected
function saveToFile(prsPath, numPath, Plabel, Rlabel, Slabel, Rfeatures)
    % Open the files for writing
    prsFile = fopen(prsPath, 'wb');
    numFile = fopen(numPath, 'wb');

    % Sanity check
    if(length(Plabel) ~= length(Rlabel) || ...
        length(Slabel) ~= length(Rlabel) || ...
        size(Rfeatures, 1) ~= length(Rlabel))

        fprintf('Inconsistency in sizes');
        error('Inconsistency in sizes: saveToFile!')
    end
        
    noInst = length(Rlabel);

    % Writing the dimension
    fprintf(numFile, '%d\n', size(Rfeatures, 2));
    % Loop and write in the format required
    for i = 1:noInst
        fprintf(prsFile, '<%s:%s:%s>\n', Plabel{i}, Rlabel{i}, Slabel{i});

        for j = 1:size(Rfeatures, 2)-1
            fprintf(numFile, '%f ', Rfeatures(i, j));
        end
        fprintf(numFile, '%f\n', Rfeatures(i, end));
    end

    % Close the files for writing
    fclose(prsFile);
    fclose(numFile);
end
