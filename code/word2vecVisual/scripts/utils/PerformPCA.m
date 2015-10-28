% This script reads the numerical features file and performs PCA
% First read the features, decide the number of dimensions, perform
% PCA and save the pca features back to file in the same format

featurePath = '/home/satwik/VisualWord2Vec/data/abstract_features_train.txt';
savePath = '/home/satwik/VisualWord2Vec/data/abstract_features_train_pca.txt';
features = dlmread(featurePath, ' ', 1, 0);

% Performance pca
[coeff, score, latent] = pca(features);

% Variance ratio to capture
ratio = 0.95;
% Number of components 
noComp = sum(cumsum(latent) < 0.95 * sum(latent));

% Writing pca features back to the file
filePt = fopen(savePath, 'wb');
% Writing the feature dimension
fprintf(filePt, '%d\n', noComp);
for i = 1:size(score, 1)
    for j = 1:noComp-1
        fprintf(filePt, '%f ', score(i, j));
    end
    fprintf(filePt, '%f\n', score(i, noComp));
end

fclose(filePt);
