CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
#CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-value

all: visualFeatures word2vectest
	$(CC) visualFeatures.o word2vectest.o -o word2vectest $(CFLAGS)
word2vectest : word2vectest.c
	$(CC) -c word2vectest.c $(CFLAGS)
visualFeatures : visualFeatures.c
	$(CC) -c visualFeatures.c $(CFLAGS)

small:
	make
	./word2vectest -train modelsNdata/coco_test.txt -output modelsNdata/word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

big:
	make
	#./word2vectest -train modelsNdata/coco_train_minus_cs_test.txt -output word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -save-vocab modelsNdata/large-vocab.txt
	./word2vectest -train modelsNdata/coco_train_minus_cs_test.txt -output word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt

newline:
	./word2vectest -train /home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_tokenized_stops.txt -output word2vec_output_newline.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

clean:
	rm -rf word2vectest visualFeatures word2phrase distance word-analogy compute-accuracy
