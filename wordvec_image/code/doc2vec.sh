#this function will convert text to lowercase and will disconnect punctuation and special symbols from words

#mkdir rnnlm
#cd rnnlm
#wget http://www.fit.vutbr.cz/~imikolov/rnnlm/rnnlm-0.3e.tgz
#tar -xvf rnnlm-0.3e.tgz
#g++ -lm -O3 -march=native -Wall -funroll-loops -ffast-math -c rnnlmlib.cpp
#g++ -lm -O3 -march=native -Wall -funroll-loops -ffast-math rnnlm.cpp rnnlmlib.o -o rnnlm

#head ../train-pos.txt -n 12300 > train
#tail ../train-pos.txt -n 200 > valid
#./rnnlm -rnnlm model-pos -train train -valid valid -hidden 50 -direct-order 3 -direct 200 -class 100 -debug 2 -bptt 4 -bptt-block 10 -binary

#head ../train-neg.txt -n 12300 > train
#tail ../train-neg.txt -n 200 > valid
#./rnnlm -rnnlm model-neg -train train -valid valid -hidden 50 -direct-order 3 -direct 200 -class 100 -debug 2 -bptt 4 -bptt-block 10 -binary

#cat ../test-pos.txt ../test-neg.txt > test.txt
#awk 'BEGIN{a=0;}{print a " " $0; a++;}' < test.txt > test-id.txt
#./rnnlm -rnnlm model-pos -test test-id.txt -debug 0 -nbest > model-pos-score
#./rnnlm -rnnlm model-neg -test test-id.txt -debug 0 -nbest > model-neg-score
#paste model-pos-score model-neg-score | awk '{print $1 " " $2 " " $1/$2;}' > ../RNNLM-SCORE
#cd ..

#mkdir word2vec
#cd word2vec
#wget http://word2vec.googlecode.com/svn/trunk/word2vec.c
################### NOW LET'S ASSUME THE CODE HAS BEEN ALREADY UPDATED TO SUPPORT SENTENCE VECTORS...
#cp ../word2vec.c .
###################
#gcc word2vec.c -o word2vec -lm -pthread -O3 -march=native -funroll-loops
time ./word2vec -train 1000paragraph_id.txt -output vectors.txt -cbow 0 -size 100 -window 10 -negative 5 -hs 0 -sample 1e-4 -threads 40 -binary 0 -iter 20 -min-count 1 -sentence-vectors 1
grep '_\*' vectors.txt > sentence_vectors.txt
#wget http://www.csie.ntu.edu.tw/~cjlin/liblinear/liblinear-1.94.zip
#unzip liblinear-1.94.zip
#cd liblinear-1.94
#make
#cd ..
head sentence_vectors.txt -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > train.txt
head sentence_vectors.txt -n 50000 | tail -n 25000 | awk 'BEGIN{a=0;}{if (a<12500) printf "1 "; else printf "-1 "; for (b=1; b<NF; b++) printf b ":" $(b+1) " "; print ""; a++;}' > test.txt
./liblinear-1.94/train -s 0 train.txt model.logreg
./liblinear-1.94/predict -b 1 test.txt model.logreg out.logreg
tail -n 25000 out.logreg > SENTENCE-VECTOR.LOGREG

#cd ..
#cat RNNLM-SCORE | awk ' \
#BEGIN{cn=0; corr=0;} \
#{ \
#  if ($3<1) if (cn<12500) corr++; \
#  if ($3>1) if (cn>=12500) corr++; \
#  cn++; \
#} \
#END{print "RNNLM accuracy: " corr/cn*100 "%";}'

cat SENTENCE-VECTOR.LOGREG | awk ' \
BEGIN{cn=0; corr=0;} \
{ \
  if ($2>0.5) if (cn<12500) corr++; \
  if ($2<0.5) if (cn>=12500) corr++; \
  cn++; \
} \
END{print "Sentence vector + logistic regression accuracy: " corr/cn*100 "%";}'

#paste RNNLM-SCORE SENTENCE-VECTOR.LOGREG | awk ' \
#BEGIN{cn=0; corr=0;} \
#{ \
#  if (($3-1)*7+(0.5-$5)<0) if (cn<12500) corr++; \
#  if (($3-1)*7+(0.5-$5)>0) if (cn>=12500) corr++; \
#  cn++; \
#} \
#END{print "FINAL accuracy: " corr/cn*100 "%";}'
