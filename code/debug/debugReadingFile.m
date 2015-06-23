% Script to debug the reading part of the new implementation
% Testing if all the values are same
load('/home/satwik/VisualWord2Vec/original/originalCode.mat');
for i = 1:length(Plabel)
    if(~strcmp(Plabel{i}, P_label{i}))
        fprintf('%d : %s %s\n', i, Plabel{i}, P_label{i});
    end
    if(~strcmp(Slabel{i}, S_label{i}))
        fprintf('%d : %s %s\n', i, Slabel{i}, S_label{i});
    end
    if(~strcmp(Rlabel{i}, R_label{i}))
        fprintf('%d : %s %s\n', i, Rlabel{i}, R_label{i});
    end
end
