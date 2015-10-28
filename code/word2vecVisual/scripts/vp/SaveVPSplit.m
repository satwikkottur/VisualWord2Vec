% Script to save the ground truth
load('../../data/visual_paraphrasing/split.mat');

totalInst = length(testind) + length(trainind);
isTrain = zeros(totalInst, 1);
isTrain(trainind) = 1;

savePath = '../dumps/vp_split.txt';
saveId = fopen(savePath, 'wb');

for i = 1:totalInst
    fprintf(saveId, '%d\n', isTrain(i));
end

fclose(saveId);
