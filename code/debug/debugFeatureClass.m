% Script to check the class of the numerical features

featurePath = '/home/satwik/VisualWord2Vec/data/features/';
files = dir(fullfile(featurePath, '*.mat'));

for i = 1:length(files)
    load(fullfile(featurePath, files(i).name));
    length(feat)
    if(~isequal(unique(feat), [0, 1])) 
        error('Error class\n');
    end
end
