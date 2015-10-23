% Script to save the ground truth
load('../../data/visual_paraphrasing/dataset.mat');

savePath = '../dumps/vp_ground_truth.txt';
saveId = fopen(savePath, 'wb');

for i = 1:length(gt)
    fprintf(saveId, '%d\n', gt(i));
end

fclose(saveId);
