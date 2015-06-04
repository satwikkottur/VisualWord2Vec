function vec=embed_str(word,dict,vec_table)

[a,b]=strsplit(word,'''s');
words1=[a;horzcat(b,{''})];
words1=words1(:);

words2={};
for i=1:length(words1)
	words2=[words2;strsplit(words1{i},{' ',',','.','/','\\','?','!'})'];
end

words3={};
for i=1:length(words2)
	if ~isequal(words2{i},'')
		words3=horzcat(words3,lower(words2{i}));
	end
end

a=zeros(length(words3),1);
b=zeros(length(words3),1);
for i=1:length(words3)
	[a(i) b(i)]=ismember(words3{i},dict);
end

if sum(a)>0
	vec=sum(vec_table(b(a>0),:),1);
else
	vec=vec_table(1,:)*0;
end
