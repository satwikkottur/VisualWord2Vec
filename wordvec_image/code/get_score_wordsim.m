function [] = get_score_wordsim()

vector_dir = '~/research/wordvec/data/1m/vectors_ling/';
out_dir = '~/research/wordvec/wordsim/1m/ling/';
num_sim = 203;
num_rel = 252;
vector_list = dir(fullfile(vector_dir,'*.bin'));
% fid = fopen('~/research/wordvec/result1_sim_1m.txt','w');
% for i = 1:length(vector_list)
%     vector_file = [vector_dir vector_list(i).name];
%     file_sim = [out_dir 'sim_' vector_list(i).name];
%     cmd_sim = ['./cal_cosdist ' vector_file ' ~/research/wordvec/wordsim/wordsim_similarity_goldstandard.txt ' file_sim]
%     unix(cmd_sim)
%     scores = csvread(file_sim);
%     x = scores(:,1);
%     y = scores(:,2);
%     [r,t,p]=spear(x,y);
%     seen_sim = size(x,1)/num_sim;
%     fprintf(fid,'vec: %s r: %f, t: %f, p: %f seen %f\n', vector_list(i).name, r, t, p, seen_sim);
% end
% fclose(fid);

fid = fopen('~/research/wordvec/ling_sim_1m.txt','w');
for i = 1:length(vector_list)
    vector_file = [vector_dir vector_list(i).name];
    file_sim = [out_dir 'sim_' vector_list(i).name];
    cmd_sim = ['./cal_cosdist ' vector_file ' ~/research/wordvec/wordsim/wordsim_similarity_goldstandard.txt ' file_sim]
    unix(cmd_sim)
    scores = csvread(file_sim);
    x = scores(:,1);
    y = scores(:,2);
    [r,t,p]=spear(x,y);
    seen_sim = size(x,1)/num_sim;
    fprintf(fid,'vec: %s r: %f, t: %f, p: %f seen %f\n', vector_list(i).name, r, t, p, seen_sim);
end
fclose(fid);
end