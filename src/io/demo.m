psrFeaturePath = '/home/satwik/VisualWord2Vec/data/PSR_features.txt';
fileId = fopen(psrFeaturePath, 'rb');

% Match with any number of any characters which are not (<>,:)
regExp = '[^<>:]*';
data = textscan(fileId, '%s', 'delimiter', '\n');
data = regexp(data{1}, regExp, 'match');

p = cellfun(@(v) v(1), data);
s = cellfun(@(v) v(2), data);
r = cellfun(@(v) v(3), data);
