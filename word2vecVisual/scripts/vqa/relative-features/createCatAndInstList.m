clear;clc;
addpath('jsonlab');
tdata=loadjson('items.json');
scene = tdata;
instancelist = cell(length(scene),1);
for i=1:1:length(scene)
    instancelist{i} = scene{i}{1};
end
instancematrix = zeros(length(instancelist),10);
count = 31;
for i=31:1:length(scene)
    numtype = scene{i}{3};
    for j=1:1:numtype
        instancematrix(i,j) = count;
        count = count+1;
    end
end
categorylist = {'human';'animal';'largeObject';'smallObject'};
save('Lists.mat','categorylist','instancelist','instancematrix','-v7.3');
disp('saved');
