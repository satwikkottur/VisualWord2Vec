% Script to analyze the clusters initially formed

% Read the file
clustIdPath = '../modelsNdata/cluster_id_save.txt';
cIds = dlmread(clustIdPath);

% Read the association file
dumpPath = '../modelsNdata/trainListings.mat';
iccv = load(dumpPath);

% Get the top 25 images for each cluster
noTop = 5;
noClusters = max(cIds(:, 1));

topMembers = zeros(noClusters, noTop);
for k = 1:noClusters
    members = find(cIds(:, 1) == k);
    [~, topIds] = sort(cIds(members, 2));

    % Account for clusters having fewer number of elements
    noPresent = min(noTop, length(members));
    topMembers(k, 1:noPresent) = members(topIds(1:noPresent));
end

% Display the top 5 images for each cluster
for k = 1:noClusters
    dispImg = [];
    % Display only if noTop exists, else a very small cluster
    if(~any(topMembers(k, :) < 1))
        % Fetch the images
        for i = 1:noTop
            img = imread();

        end
    end
end
