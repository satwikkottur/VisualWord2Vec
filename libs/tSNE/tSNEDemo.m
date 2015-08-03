% A demo script to understand the funtionality and usage of tSNE in matlab
% Source: http://lvdmaaten.github.io/tsne/User_guide.pdf 

% Load data
%load ’mnist_train.mat’
%ind = randperm(size(train_X, 1));
%train_X = train_X(ind(1:5000),:);
%train_labels = train_labels(ind(1:5000));
% Set parameters
no_dims = 2;
initial_dims = 50;
perplexity = 30;
no_points = 100;

% Generate random data for visualization
train_X = rand(no_points, initial_dims);
train_labels = randi([1 3], no_points, 1);

% Run t−SNE
mappedX = tsne(train_X, train_labels, no_dims, initial_dims, perplexity);

% Plot results
gscatter(mappedX(:,1), mappedX(:,2), train_labels);
