CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
#CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-value

all: word2vectest word2phrase distance word-analogy compute-accuracy

word2vectest : word2vectest.c
	$(CC) word2vectest.c -o word2vectest $(CFLAGS)
word2phrase : word2phrase.c
	$(CC) word2phrase.c -o word2phrase $(CFLAGS)
distance : distance.c
	$(CC) distance.c -o distance $(CFLAGS)
word-analogy : word-analogy.c
	$(CC) word-analogy.c -o word-analogy $(CFLAGS)
compute-accuracy : compute-accuracy.c
	$(CC) compute-accuracy.c -o compute-accuracy $(CFLAGS)
	chmod +x *.sh

small:
	./word2vectest -train coco_test.txt -output word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

big:
	./word2vectest -train coco_train_minus_cs_test.txt -output word2vec_output.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

clean:
	rm -rf word2vectest word2phrase distance word-analogy compute-accuracy
