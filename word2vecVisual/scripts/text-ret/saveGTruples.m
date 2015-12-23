% Read the matfiles and save the ground tuples
dumpPath = '../modelsNdata/trainListings.mat';
matPath = '/home/satwik/VisualWord2Vec/data/rawdata/';
load(dumpPath)

savePath = '/home/satwik/VisualWord2Vec/data/text-retrieval/pilot_gt.txt';
saveId = fopen(savePath, 'wb');

for i = 1:length(listing)
    fullPath = fullfile(matPath, listing(i).name);
    tag = strrep(listing(i).name, '.mat', '');

    data = load(fullPath);

    % Save the tag vs relation primary secondary
    fprintf(saveId, '%s:<%s:%s:%s>\n', tag, data.relationName, ...
                                data.primaryName, data.secondaryName);
end

fclose(saveId);
