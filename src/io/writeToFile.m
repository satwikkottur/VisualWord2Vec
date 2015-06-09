function writeToFile(dataPath, prsFeaturePath, numFeaturePath)
    % Function to read the MAT files and write to text files
    % We use the following format:
    % <PrimaryName, SecondaryName, Relation>:<feature>
    %
    % First clause is read from : /home/satwik/VisualWord2Vec/data/rawdata
    % Next clause is read from : /home/satwik/VisualWord2Vec/data/features
    %
    % Input:
    % dataPath = Path to the data folder where rawdata/ and features/ folders are 
    %           present
    % prsFeaturePath = Path where PRS feature file must be saved
    % numFeaturePath = path where numerical feature file must be saved

    prsId = fopen(prsFeaturePath, 'wb');
    numId = fopen(numFeaturePath, 'wb');

    listing = dir(fullfile(dataPath, 'rawdata', '*.mat'));
    noLines = length(listing);

    fprintf(prsId, '%d\n', noLines);

    for i = 1:noLines
        filePath = fullfile(dataPath, 'rawdata', listing(i).name);
        data = load(filePath);

        % Writing the first clause
        fprintf(prsId, '<%s:%s:%s>\n', data.primaryName, ...
                                            data.secondaryName, data.relationName);
        
        filePath = fullfile(dataPath, 'features', ...
                        [listing(i).name(1:end-5), '.mat']);
        featureData = open(filePath);
        
        % Writing the feature
        fprintf(numId, '%d ', featureData.feat);
        fprintf(numId, '\n');
        fprintf('Iteration: %d\n', i);
    end

    fclose(numId);
    fclose(prsId);
end
