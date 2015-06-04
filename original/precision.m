function [MAP base]=precision(score,label)

ind=randperm(length(score),length(score));
score=score(ind);
label=label(ind);

[a b]=sort(score,'descend');
[junk binv]=sort(b);

rank_pos=binv(label==1);
rank_pos=sort(rank_pos);

MAP=mean([1:length(rank_pos)]./rank_pos');
base=sum(label==1)/length(label);

[1:length(rank_pos)]./rank_pos';