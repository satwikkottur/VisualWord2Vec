% Debugging validation and training set loads
tic
load('/home/satwik/VisualWord2Vec/original/trainValLoad.mat');
fprintf('Testing validation and testing set loads\n');

assert(any(size(test_P) == size(testP)))
assert(any(size(test_R) == size(testR)))
assert(any(size(test_S) == size(testS)))
assert(any(size(test_P_unique_label) == size(testPdict)))
assert(any(size(test_R_unique_label) == size(testRdict)))
assert(any(size(test_S_unique_label) == size(testSdict)))
assert(any(size(testPembed) == size(test_P_embedding)))
assert(any(size(testRembed) == size(test_R_embedding)))
assert(any(size(testSembed) == size(test_S_embedding)))

assert(any(size(val_P) == size(valP)))
assert(any(size(val_R) == size(valR)))
assert(any(size(val_S) == size(valS)))
assert(any(size(val_P_unique_label) == size(valPdict)))
assert(any(size(val_R_unique_label) == size(valRdict)))
assert(any(size(val_S_unique_label) == size(valSdict)))
assert(any(size(valPembed) == size(val_P_embedding)))
assert(any(size(valRembed) == size(val_R_embedding)))
assert(any(size(valSembed) == size(val_S_embedding)))

for i = 1:length(test_P)
    if(~strcmp(test_P{i}, testP{i}))
        i
    end
end

for i = 1:length(test_R)
    if(~strcmp(test_R{i}, testR{i}))
        i
    end
end

for i = 1:length(test_S)
    if(~strcmp(test_S{i}, testS{i}))
        i
    end
end

for i = 1:length(val_P)
    if(~strcmp(val_P{i}, valP{i}))
        i
    end
end

for i = 1:length(val_R)
    if(~strcmp(val_R{i}, valR{i}))
        i
    end
end

for i = 1:length(val_S)
    if(~strcmp(val_S{i}, valS{i}))
        i
    end
end

for i = 1:length(test_P_unique_label)
    if(~strcmp(test_P_unique_label{i}, testPdict{i}))
        i
    end
end

for i = 1:length(test_R_unique_label)
    if(~strcmp(test_R_unique_label{i}, testRdict{i}))
        i
    end
end

for i = 1:length(test_S_unique_label)
    if(~strcmp(test_S_unique_label{i}, testSdict{i}))
        i
    end
end

for i = 1:length(val_P_unique_label)
    if(~strcmp(val_P_unique_label{i}, valPdict{i}))
        i
    end
end

for i = 1:length(val_R_unique_label)
    if(~strcmp(val_R_unique_label{i}, valRdict{i}))
        i
    end
end

for i = 1:length(val_S_unique_label)
    if(~strcmp(val_S_unique_label{i}, valSdict{i}))
        i
    end
end

sum(sum(abs(test_S_embedding - testSembed)))
sum(sum(abs(test_R_embedding - testRembed)))
sum(sum(abs(test_P_embedding - testPembed)))

sum(sum(abs(val_S_embedding - valSembed)))
sum(sum(abs(val_R_embedding - valRembed)))
sum(sum(abs(val_P_embedding - valPembed)))

time = toc;
fprintf('Successfully validation and testing set loads in %f\n', time);

%who
