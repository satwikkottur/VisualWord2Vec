clear all; close all; clc;
addpath('jsonlab');
addpath('Occurrence/gmmFeat')
datapath = '/home/satwik/VisualWord2Vec/data/vqa/iccv_json';
%datapath = 'jsondata';
featpath = '/home/satwik/VisualWord2Vec/data/vqa/iccv_feature';
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
errorCausingImages = [];
count = 1;
listing = dir(datapath);
listing(1:2) = [];
for i=1:1:length(listing)
    try
        path = fullfile(datapath,listing(i).name);
        feat = ExtractFeaturesForEachImage(path, catlist, inslist, insmat, humandata, GAbsPos, GRelPos);
        filename = listing(i).name;
        filename = filename(1:end-5);
        filename = [filename '.mat'];
        save(fullfile(featpath,filename),'feat');
        clear feat; clear path; clear filename;
        disp(['Saved Feature for Image ' num2str(i) '/' num2str(length(listing))]);
    catch
        errorCausingImages(count) = i;
        count = count+1;
    end
end
if ~isempty(errorCausingImages)
    disp('****************************************');
    disp('The indices of error causing images are:');
    disp(errorCausingImages);
end
