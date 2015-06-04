
function [model_test model_crossval acc_crossval random_crossval]=perclass_libsvm(label,data,c,nfolds)
%per-class SVM
nim=size(label,1);
acc_crossval=zeros(size(label,2),nfolds);
random_crossval=zeros(size(label,2),nfolds);
model_crossval=cell(size(label,2),nfolds);
model_test=cell(size(label,2),1);
for i=1:size(label,2)
	for foldid=1:nfolds
		fold_ind=foldid:nfolds:nim;
		fold_ind_inv=setdiff(1:nim,fold_ind);
		
		model_crossval{i,foldid}=svmtrain(label(fold_ind_inv,i),data(fold_ind_inv,:),['-t 0 -h 0 -c ' num2str(c)]);
		if length(model_crossval{i,foldid}.Label)==2
			if(model_crossval{i,foldid}.Label(1)~=1)
				print('something wrong!');
			end
			[~,~,scores_cval]=svmpredict(label(fold_ind,i),data(fold_ind,:),model_crossval{i,foldid});
			[acc_crossval(i,foldid),random_crossval(i,foldid)]=precision(scores_cval,label(fold_ind,i));
		else
			acc_crossval(i,foldid)=NaN;
			random_crossval(i,foldid)=0;
		end
	end
	model_test{i}=train(label(:,i),sparse(data(:,:)),['-c ' num2str(c)]);
end


tmp=[mean(acc_crossval');mean(random_crossval')]
tmp2=tmp(:,~isnan(tmp(1,:)))
mean(tmp2')
