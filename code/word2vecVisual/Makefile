CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
#CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-value
MOVE=mv *.o $(BIN)
BIN = bin/

#all: visualFeatures word2vec
#	$(CC) visualFeatures.o word2vec.o -o word2vec $(CFLAGS)
all: word2vec visualFeatures
	$(MOVE)
	$(CC) $(BIN)word2vec.o $(BIN)visualFeatures.o -o $(BIN)word2vec $(CFLAGS)
word2vec : word2vec.c
	$(CC) -c word2vec.c $(CFLAGS)
visualFeatures : visualFeatures.c
	$(CC) -c visualFeatures.c $(CFLAGS)

run:
	make
	./$(BIN)word2vec -train modelsNdata/coco_test.txt -output word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt

small:
	make
	./$(BIN)word2vec -train modelsNdata/coco_test.txt -output modelsNdata/word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

big:
	make
	#./word2vec -train modelsNdata/coco_train_minus_cs_test.txt -output modelsNdata/word2vec_after_refine.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0
	./$(BIN)word2vec -train modelsNdata/coco_train_minus_cs_test.txt -output modelsNdata/word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt

newline:
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_tokenized_stops.txt -output word2vec_output_newline.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

clean:
	rm -rf word2vec visualFeatures *.o bin/*
