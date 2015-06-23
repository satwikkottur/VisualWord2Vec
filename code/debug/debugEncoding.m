% Script to debug one hot encoding of the words
load('/home/satwik/VisualWord2Vec/original/originalCode.mat');

diff = abs(Pencoding - P_label_01);
sum(diff(:))

diff = abs(Rencoding - R_label_01);
sum(diff(:))

diff = abs(Sencoding - S_label_01);
sum(diff(:))

assert(length(Pdict) == length(P_unique_label))
assert(length(Sdict) == length(S_unique_label))
assert(length(Rdict) == length(R_unique_label))

for i = 1:length(Pdict)
    if(~strcmp(Pdict(i), P_unique_label(i)))
        i
    end
end

for i = 1:length(Sdict)
    if(~strcmp(Sdict(i), S_unique_label(i)))
        i
    end
end
for i = 1:length(Rdict)
    if(~strcmp(Rdict(i), R_unique_label(i)))
        i
    end
end
