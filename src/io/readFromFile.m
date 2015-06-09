function[Plabel, Slabel, Rlabel, Rfeatures] =  readFromFile(psrFeaturePath, numFeaturePath)
    % Function to read features from a text file
    % We use the following format:
    % <PrimaryName, SecondaryName, Relation>:<feature>
    %
    % First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
    % Next clause is read from : /home/satwik/VisualWord2Vec/data/features
    % 
    % Input:
    % psrFeaturePath = Path to the PSR feature file
    % numFeaturePath = Path to the numerical features
    % 
    % Output:
    % Plabel = P features
    % Slabel = S features
    % Rlabel = R features
    % Rfeatures = Numerical features

    if(~exist(psrFeaturePath, 'file'))
        error('File does not exist\n')
    end
    fileId = fopen(psrFeaturePath, 'rb');

    % Reading from the file
    noLines = str2num(fgetl(fileId));

    % Initializing the structures
    Plabel = cell(noLines, 1);
    Slabel = cell(noLines, 1);
    Rlabel = cell(noLines, 1);
    R_feature = cell(noLines, 1);

    % Match with any number of any characters which are not (<>,:)
    regExp = '[^<>:]*';
    lineId = 1;

    % Reading the first line for PRS features
    tline = fgetl(fileId);
    while ischar(tline)
        splitString = regexp(tline, regExp, 'match');
        assert(length(splitString) == 3);

        % Reading the (P,S,R) labels
        Plabel{lineId} = splitString{1};
        Slabel{lineId} = splitString{2};
        Rlabel{lineId} = splitString{3};

        lineId = lineId + 1;

        % Read the next line
        tline = fgetl(fileId);
    end

    % Closing the file
    fclose(fileId);

    % Reading numerical features 
    if(~exist(numFeaturePath, 'file'))
        error('File does not exist\n')
    end
    Rfeatures = dlmread(numFeaturePath, ' ');
    % Ignore the last column as it is due to additional space at the end
    Rfeatures = Rfeatures(:, 1:end-1);
end
