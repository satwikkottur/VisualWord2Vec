% Script to save the ground truth
if(~exist('gt', 'var'))
    load('../../data/visual_paraphrasing/dataset.mat');
end
load('../../data/visual_paraphrasing/split.mat');

% Get the ratio for the training inds
%gtTrain = gt(trainind);
%sum(gtTrain)./length(gtTrain)
% 1 from gt = 1
% 2 from gt = 0
% Only from traininds
gtInds = find(gt == 1);
gtTrainInds = intersect(gtInds, trainind);

falseInds = find(gt == 0);
falseTrainInds = intersect(falseInds, trainind);

% Choose 1k from gtTrainInds and 2k from falseTrainInds
noVals = 1000;
gtValInds = datasample(gtTrainInds, floor(noVals/3), 'Replace', false);
falseValInds = datasample(falseTrainInds, 2* floor(noVals/3), 'Replace', false);

valInds = union(gtValInds, falseValInds);

valSet = zeros(length(gt), 1);
valSet(valInds) = 1;

savePath = '../dumps/vp_val_inds_1k.txt';
saveId = fopen(savePath, 'wb');

for i = 1:length(gt)
    fprintf(saveId, '%d\n', valSet(i));
end

%fclose(saveId);
