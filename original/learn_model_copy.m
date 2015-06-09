%
%datapath = '/home/linxiao/public_html/browser_sentint_full4/image_json';
%featpath = '/home/linxiao/public_html/browser_sentint_full4/image_features_tanmay';
%%featpath_pairs = '/home/linxiao/public_html/browser_sentint_full4/image_features_pairs';
%%word2vec model. change to enwiki9.mat for wikipedia word2vec
%w2v_model='coco_w2v.mat';
%%C values and thresholds need to be tuned manually, check "TODO"s
%
%
%%Collect PRS and features
%listing = dir(fullfile(datapath,'*.mat'));
%R_label=cell(length(listing),1);
%P_label=cell(length(listing),1);
%S_label=cell(length(listing),1);
%features=cell(length(listing),1);
%
%%for each image grab its PRS
%for i=1:1:length(listing)
%    fname_data = fullfile(datapath,listing(i).name);
%    fname_feature=listing(i).name;
%    fname_feature = fullfile(featpath,[fname_feature(1:end-5) '.mat']);
%    
%    data=load(fname_data);
%    P_label{i}=data.primaryName;
%    S_label{i}=data.secondaryName;
%    R_label{i}=data.relationName;
%    
%    %fprintf('%s  : %s : %s\n', P_label{i}, S_label{i}, R_label{i});
%    feat=load(fname_feature);
%    features{i}=feat.feat;
%    fprintf('Iteration : %d / %d\n', i, length(listing));
%end
%
% Saving the MAT file
%save('originalCode.mat');

% Loading the MAT file
load('originalCode.mat');

addpath(genpath('./'));
%R labels
nim=length(features);
R_unique_label=unique(R_label);
[~ , R_id]=ismember(R_label,R_unique_label);
R_label_01=zeros(nim,length(R_unique_label));
for i=1:nim
    R_label_01(i,R_id(i))=1;
end
R_label_01=2*R_label_01-1;
%P labels
P_unique_label=unique(P_label);
P_embedding=cell(length(P_unique_label),1);
[~ , P_id]=ismember(P_label,P_unique_label);
P_label_01=zeros(nim,length(P_unique_label));
for i=1:nim
    P_label_01(i,P_id(i))=1;
end
P_label_01=2*P_label_01-1;

%S labels
S_unique_label=unique(S_label);
S_embedding=cell(length(S_unique_label),1);
[~ , S_id]=ismember(S_label,S_unique_label);
S_label_01=zeros(nim,length(S_unique_label));
for i=1:nim
    S_label_01(i,S_id(i))=1;
end
S_label_01=2*S_label_01-1;

%features
ndims=length(features{1});
R_feat=zeros(nim,ndims);
for i=1:nim
    R_feat(i,:)=features{i};
end
R_feat=double(R_feat);

save('originalCode.mat');
return
w2v=load(w2v_model);


%compute PRS word2vec embeddings
R_unique_label=unique(R_label);
R_embedding=cell(length(R_unique_label),1);
for i=1:length(R_unique_label)
    disp(num2str(i));
    R_embedding{i}=embed_str(R_unique_label{i},w2v.tokens,w2v.fv);
end
R_embedding=cell2mat(R_embedding);
for i=1:length(P_unique_label)
    disp(num2str(i));
    P_embedding{i}=embed_str(P_unique_label{i},w2v.tokens,w2v.fv);
end
P_embedding=cell2mat(P_embedding);
S_unique_label=unique(S_label);
S_embedding=cell(length(S_unique_label),1);
for i=1:length(S_unique_label)
    disp(num2str(i));
    S_embedding{i}=embed_str(S_unique_label{i},w2v.tokens,w2v.fv);
end
S_embedding=cell2mat(S_embedding);


%model2: embedding models
nfolds=5;
addpath(genpath('./'));

%TODO: don't preload when you want to train your own model.
if 1
    load workspacedump_w_models_coco.mat;
else
    %train models
    [R_model_test_embed R_model_crossval_embed R_acc_crossval_embed R_random_crossval_embed]=embedding(R_label_01,R_embedding,R_feat,[0.0001 0.001 0.01 0.1],nfolds,12780,100)
    [P_model_test_embed P_model_crossval_embed P_acc_crossval_embed P_random_crossval_embed]=embedding(P_label_01,P_embedding,R_feat,[0.0001 0.001 0.01 0.1],nfolds,12780,100)
    [S_model_test_embed S_model_crossval_embed S_acc_crossval_embed S_random_crossval_embed]=embedding(S_label_01,S_embedding,R_feat,[0.0001 0.001 0.01 0.1],nfolds,12780,100)
    %find the best C based on *_acc_crossval_embed (Here I'm fixing to 0.01), and turn w into matrix for efficient score computation
    R_A=reshape(R_model_test_embed{3}.w,[ndims,200]);
    P_A=reshape(P_model_test_embed{3}.w,[ndims,200]);
    S_A=reshape(S_model_test_embed{3}.w,[ndims,200]);
end

%save('kron_models_new.mat','R_model_test_embed','R_model_crossval_embed', 'R_acc_crossval_embed', 'R_random_crossval_embed','P_model_test_embed', 'P_model_crossval_embed', 'P_acc_crossval_embed', 'P_random_crossval_embed','S_model_test_embed', 'S_model_crossval_embed', 'S_acc_crossval_embed', 'S_random_crossval_embed','R_A','P_A','S_A');

%model1: directly classify. Not used here.
%nfolds=5;
%c=0.1;
%[R_model_test R_model_crossval R_acc_crossval R_random_crossval]=perclass(R_label_01,R_feat,c,nfolds)
%[P_model_test P_model_crossval P_acc_crossval P_random_crossval]=perclass(P_label_01,R_feat,c,nfolds)
%[S_model_test S_model_crossval S_acc_crossval S_random_crossval]=perclass(S_label_01,R_feat,c,nfolds)

%load val and test data

val=load('val.mat');
test=load('test.mat');
nval=size(val.data,1);
ntest=size(test.data,1);

val_P=cell(nval,1);
val_R=cell(nval,1);
val_S=cell(nval,1);
for i=1:nval
    %trim off the strange spaces during python to matlab conversion
    val_P{i}=strtrim(val.data{i,1}(1,:));
    val_R{i}=strtrim(val.data{i,1}(2,:));
    val_S{i}=strtrim(val.data{i,1}(3,:));
end

test_P=cell(ntest,1);
test_R=cell(ntest,1);
test_S=cell(ntest,1);
for i=1:ntest
    %trim off the strange spaces during python to matlab conversion
    test_P{i}=strtrim(test.data{i,1}(1,:));
    test_R{i}=strtrim(test.data{i,1}(2,:));
    test_S{i}=strtrim(test.data{i,1}(3,:));
end

%index and get word2vec embeddings for val and test PRS
val_R_unique_label=unique(val_R);
val_R_embedding=cell(length(val_R_unique_label),1);
[~ , val_R_id]=ismember(val_R,val_R_unique_label);
for i=1:length(val_R_unique_label)
    disp(num2str(i));
    val_R_embedding{i}=embed_str(val_R_unique_label{i},w2v.tokens,w2v.fv);
end
val_R_embedding=cell2mat(val_R_embedding);

val_P_unique_label=unique(val_P);
val_P_embedding=cell(length(val_P_unique_label),1);
[~ , val_P_id]=ismember(val_P,val_P_unique_label);
for i=1:length(val_P_unique_label)
    disp(num2str(i));
    val_P_embedding{i}=embed_str(val_P_unique_label{i},w2v.tokens,w2v.fv);
end
val_P_embedding=cell2mat(val_P_embedding);

val_S_unique_label=unique(val_S);
val_S_embedding=cell(length(val_S_unique_label),1);
[~ , val_S_id]=ismember(val_S,val_S_unique_label);
for i=1:length(val_S_unique_label)
    disp(num2str(i));
    val_S_embedding{i}=embed_str(val_S_unique_label{i},w2v.tokens,w2v.fv);
end
val_S_embedding=cell2mat(val_S_embedding);

%save('embedding_val.mat','val_R_embedding','val_P_embedding','val_S_embedding','val_R_id','val_P_id','val_S_id');

test_R_unique_label=unique(test_R);
test_R_embedding=cell(length(test_R_unique_label),1);
[~ , test_R_id]=ismember(test_R,test_R_unique_label);
for i=1:length(test_R_unique_label)
    disp(num2str(i));
    test_R_embedding{i}=embed_str(test_R_unique_label{i},w2v.tokens,w2v.fv);
end
test_R_embedding=cell2mat(test_R_embedding);

test_P_unique_label=unique(test_P);
test_P_embedding=cell(length(test_P_unique_label),1);
[~ , test_P_id]=ismember(test_P,test_P_unique_label);
for i=1:length(test_P_unique_label)
    disp(num2str(i));
    test_P_embedding{i}=embed_str(test_P_unique_label{i},w2v.tokens,w2v.fv);
end
test_P_embedding=cell2mat(test_P_embedding);

test_S_unique_label=unique(test_S);
test_S_embedding=cell(length(test_S_unique_label),1);
[~ , test_S_id]=ismember(test_S,test_S_unique_label);
for i=1:length(test_S_unique_label)
    disp(num2str(i));
    test_S_embedding{i}=embed_str(test_S_unique_label{i},w2v.tokens,w2v.fv);
end
test_S_embedding=cell2mat(test_S_embedding);

%save('embedding_test.mat','test_R_embedding','test_P_embedding','test_S_embedding','test_R_id','test_P_id','test_S_id');

%convert human scores to labels
val_label=zeros(nval,1);
for i=1:nval
    val_label(i)=val.data{i,3};
end
val_score=val_label;
val_label=val_label>0;

test_label=zeros(ntest,1);
for i=1:ntest
    test_label(i)=test.data{i,3};
end
test_score=test_label;
test_label=test_label>0;

%Visual model
%compute x'A for each x
R_score_test_embed=R_feat*R_A;
P_score_test_embed=R_feat*P_A;
S_score_test_embed=R_feat*S_A;

%compute x'Ay for each x and y in val
val_R_unique_score_embed=R_score_test_embed*val_R_embedding';
val_P_unique_score_embed=P_score_test_embed*val_P_embedding';
val_S_unique_score_embed=S_score_test_embed*val_S_embedding';

%compute x'Ay for each x and y in test
test_R_unique_score_embed=R_score_test_embed*test_R_embedding';
test_P_unique_score_embed=P_score_test_embed*test_P_embedding';
test_S_unique_score_embed=S_score_test_embed*test_S_embedding';

%TODO: manually change threshold until prec_val is maximized. Visual threshold usually around 0~1
threshold=0.6;
visual_val_score=zeros(nval,1);
for i=1:nval
    disp(num2str(i));
    visual_val_score(i,1)=mean(max(val_P_unique_score_embed(:,val_P_id(i))+val_R_unique_score_embed(:,val_R_id(i))+val_S_unique_score_embed(:,val_S_id(i))-threshold,0),1);
end
[prec_val,base_val]=precision(visual_val_score,val_label)

%Then check test performance with this threshold
visual_test_score=zeros(ntest,1);
for i=1:ntest
    disp(num2str(i));
    visual_test_score(i,1)=mean(max(test_P_unique_score_embed(:,test_P_id(i))+test_R_unique_score_embed(:,test_R_id(i))+test_S_unique_score_embed(:,test_S_id(i))-threshold,0),1);
end
[prec,base]=precision(visual_test_score,test_label)


%Text model
%compute cosine similarity-1 for val
val_R_unique_score_embed_text=-pdist2(R_embedding(R_id,:),val_R_embedding,'cosine');
val_P_unique_score_embed_text=-pdist2(P_embedding(P_id,:),val_P_embedding,'cosine');
val_S_unique_score_embed_text=-pdist2(S_embedding(S_id,:),val_S_embedding,'cosine');

%compute cosine similarity-1 for test
test_R_unique_score_embed_text=-pdist2(R_embedding(R_id,:),test_R_embedding,'cosine');
test_P_unique_score_embed_text=-pdist2(P_embedding(P_id,:),test_P_embedding,'cosine');
test_S_unique_score_embed_text=-pdist2(S_embedding(S_id,:),test_S_embedding,'cosine');


%TODO: manually change threshold until prec_val is maximized. Text threshold usually around -2~1
threshold=-1.2;
text_val_score=zeros(nval,1);
for i=1:nval
    disp(num2str(i));
    text_val_score(i,1)=mean(max(val_P_unique_score_embed_text(:,val_P_id(i))+val_R_unique_score_embed_text(:,val_R_id(i))+val_S_unique_score_embed_text(:,val_S_id(i))-threshold,0),1);
end
[prec_val,base_val]=precision(text_val_score,val_label)


text_test_score=zeros(ntest,1);
for i=1:ntest
    disp(num2str(i));
    text_test_score(i,1)=mean(max(test_P_unique_score_embed_text(:,test_P_id(i))+test_R_unique_score_embed_text(:,test_R_id(i))+test_S_unique_score_embed_text(:,test_S_id(i))-threshold,0),1);
end
[prec,base]=precision(text_test_score,test_label)



%Combining Text+Visual
%TODO: Choose your feature combination
hybrid_feat_val=[text_val_score visual_val_score];
hybrid_feat_test=[text_test_score visual_test_score];

%TODO: Tune C till optimal
c=10000;
%crossval
[hybrid_model_test hybrid_model_crossval hybrid_acc_crossval hybrid_random_crossval]=perclass(val_label*2-1,hybrid_feat_val,c,5)
hybrid_perf_crossval=mean(hybrid_acc_crossval)
%test
[~,~,hybrid_score_test]=predict(test_label*2-1,sparse(hybrid_feat_test),hybrid_model_test{1});
[hybrid_perf_test, baseline_test]=precision(hybrid_score_test,test_label*2-1)
corr(hybrid_score_test, test_score, 'type', 'Spearman')
corr(hybrid_score_test, test_score, 'type', 'Kendall')



%piece of script that deal with bing
load bing_val;
bing_val=double(data);
load bing_test;
bing_test=double(data);
%TODO: Tune C till optimal
c=0.0001;
%crossval
[bing_model_test bing_model_crossval bing_acc_crossval bing_random_crossval]=perclass(val_label*2-1,log(bing_val+1),c,nfolds)
bing_perf_crossval=mean(bing_acc_crossval)
%test
[~,~,bing_score_test]=predict(test_label*2-1,sparse(log(bing_test+1)),bing_model_test{1});
[bing_perf_test,baseline_test]=precision(bing_score_test,test_label*2-1)
corr(bing_score_test, test_score, 'type', 'Spearman')
corr(bing_score_test, test_score, 'type', 'Kendall')
%Get val score of bing, potentially for hybrid models
[~,~,bing_score_val]=predict(val_label*2-1,sparse(log(bing_val+1)),bing_model_test{1});


%When test features are different in the case of semi supervised. Not used due to bad performance
if 0
    listing_pairs = dir(fullfile(datapath,'*.mat'));
    features_test=cell(length(listing),1);
    %for each image grab its PRS
    for i=1:1:length(listing)
        fname_feature=listing(i).name;
        fname_feature = fullfile(featpath_pairs,[fname_feature(1:end-5) '.mat']);
        
        fv=load(fname_feature);
        features_test{i}=cell2mat(fv.feat);
    end

    for i=1:1:length(listing)
        features_test{i}=cell2mat(features_test{i});
    end

    R_feat_test=double(cell2mat(features_test));



    P_score_test=cell(1,size(P_label_01,2));
    for i=1:size(P_label_01,2)
        [~,~,P_score_test{i}]=predict(ones(size(R_feat_test,1),1),sparse(R_feat_test),P_model_test{i});
    end
    P_score_test=cell2mat(P_score_test);


    S_score_test=cell(1,size(S_label_01,2));
    for i=1:size(S_label_01,2)
        [~,~,S_score_test{i}]=predict(ones(size(R_feat_test,1),1),sparse(R_feat_test),S_model_test{i});
    end
    S_score_test=cell2mat(S_score_test);
    
end
