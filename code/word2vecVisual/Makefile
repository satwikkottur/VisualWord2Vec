CC = @gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
#CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

# Liblinear path
LINEARPATH=/home/satwik/VisualWord2Vec/libs/liblinear-2.1/
LIBLINEAR=$(LINEARPATH)linear.o $(LINEARPATH)tron.o $(LINEARPATH)blas/blas.a
BLAS=$(LINEARPATH)blas/blas.a

YAEL = -I /home/satwik/VisualWord2Vec/libs/yael -L /home/satwik/VisualWord2Vec/libs/yael/yael -lyael

CFLAGS = -lm -pthread -O3 -march=native -Wall -lstdc++ -funroll-loops -Wno-unused-value $(YAEL) 

MOVE=mv *.o $(BIN)
BIN = bin/

NAMES = word2vec.o	 \
	visualFeatures.o \
	debugFunctions.o \
	vpFunctions.o	\
	helperFunctions.o\
	refineFunctions.o\
	liblinearWrapper.o\
	cocoFunctions.o

OBJ=$(NAMES:%=$(BIN)%)
SRCS=$(NAMES:%.o=%.c)

all: $(OBJ)
	$(CC) $(OBJ) -o $(BIN)word2vec $(BLAS) $(LIBLINEAR) $(CFLAGS)

$(BIN)%.o : %.c
	$(CC) -c $< -o $@ $(CFLAGS)

vp:
	make
	./$(BIN)word2vec -train modelsNdata/vp/coco_lemma_vp_lemma.txt -output modelsNdata/word2vec_vp_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

coco:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_lemma.txt -output modelsNdata/word2vec_coco_word2vec.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

wikiplain:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/enwik9 -output /home/satwik/VisualWord2Vec/models/wiki_just_embeddings.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 36 -binary 0 -min-count 0

wikivp:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/wiki_vp_train_lemma.txt -output modelsNdata/word2vec_after_refine_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

common:
	make
	./$(BIN)word2vec -train modelsNdata/coco_train_minus_cs_test.txt -output modelsNdata/word2vec_output_bestmodel_single.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt

clean:
	rm -rf word2vec visualFeatures *.o $(BIN)*
