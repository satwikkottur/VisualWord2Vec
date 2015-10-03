function embed_str(word,filePt)
fprintf(filePt, '%s = ', word);
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
apos = false;
for i=1:length(words3)
    if(strcmp(words3{i}, '''s'))
        apos = true;
    else
        fprintf(filePt, ':%s:', words3{i});
    end
	%[a(i) b(i)]=ismember(words3{i},dict);
end
if(apos)
    fprintf(filePt, ':''s:');
end
fprintf(filePt, '\n');
