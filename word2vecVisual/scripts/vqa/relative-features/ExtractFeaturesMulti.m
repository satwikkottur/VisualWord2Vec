clear all; close all; clc;
addpath('jsonlab');
addpath('Occurrence/gmmFeat')
datapath = '/home/satwik/VisualWord2Vec/data/vqa/iccv_json';
%datapath = 'jsondata';
featpath = '/home/satwik/VisualWord2Vec/data/vqa/iccv_features';
%featpath = 'features';

% get category and instance lists
f1 = load('Lists.mat');
catlist = f1.categorylist;
inslist = f1.instancelist;
insmat = f1.instancematrix;

% load the GMMs for Absolute and Relative Location Features
load('absGMM.mat');
load('relGMM.mat');

% get human data
fdata = fopen('paper_doll_data.txt');
tempread = textscan(fdata,'%s','delimiter','\n');
fclose(fdata);
humandata = zeros(20,3);
for i=1:1:length(tempread{1})
    t1 = textscan(tempread{1}{i},'%d');
    humandata(i,:) = t1{1};
end

% get features
listing = dir(datapath);
listing(1:2) = [];
noWorkers = 32;
% Launch workers
parpool('local', noWorkers);

errorCausingImages = cell(noWorkers, 1);
%path = cell(noWorkers, 1);
%feat = cell(noWorkers, 1);
%filename = cell(noWorkers, 1);
count = ones(noWorkers, 1);
parfor j = 1:noWorkers
    for i= j:noWorkers:length(listing)
        try
            path = fullfile(datapath,listing(i).name);

            % Extract the feature
            feat = ExtractFeaturesForEachImage(path, catlist, inslist, insmat, humandata, GAbsPos, GRelPos);

            filename = listing(i).name;
            filename = filename(1:end-5);
            filename = [filename '.mat'];
            parsave(fullfile(featpath, filename), feat);
            %save(fullfile(featpath,filename),'feat');
            %clear feat; clear path; clear filename;
            
            % Opening a file and write the vector
            %fullname = fullfile(featpath, filename);
            %fileId = fopen(fullname, 'w');
            %dlmwrite(fullname, feat, 'delimiter', ' ');
            %fclose(fileId);
            
            fprintf('Saved feature: %d / %d\n', i, length(listing));
        catch
            fprintf('Error for Image: %d / %d\n', i, length(listing));
            errorCausingImages{j}(count(j)) = i;
            count(j) = count(j) + 1;
        end
    end
end
if ~isempty(errorCausingImages)
    disp('****************************************');
    disp('The indices of error causing images are:');
    disp(errorCausingImages);
end
