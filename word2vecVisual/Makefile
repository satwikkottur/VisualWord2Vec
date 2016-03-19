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
	cocoFunctions.o\
	vqaFunctions.o\
	retriever.o

OBJ=$(NAMES:%=$(BIN)%)
SRCS=$(NAMES:%.o=%.c)

all: $(OBJ)
	$(CC) $(OBJ) -o $(BIN)word2vec $(BLAS) $(LIBLINEAR) $(CFLAGS)

$(BIN)%.o : %.c
	$(CC) -c $< -o $@ $(CFLAGS)

debug:
	make
	./$(BIN)word2vec -embed-path modelsNdata/al_vectors.txt -output modelsNdata/al_vectors_debug.bin -size 200 -threads 12

vqa:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/vqa/vqa_train_captions_lemma.txt -output modelsNdata/word2vec_coco_word2vec.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 30 -binary 0 -min-count 0

vp:
	make
	./$(BIN)word2vec -train modelsNdata/vp/coco_lemma_vp_lemma.txt -output modelsNdata/word2vec_vp_lemma_100hidden.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 24 -binary 0 -min-count 0

coco:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/coco-cnn/captions_coco_lemma.txt -output modelsNdata/word2vec_coco_after_300.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 30 -binary 0 -min-count 0

wikiplain:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/enwik9 -output /home/satwik/VisualWord2Vec/models/wiki_just_embeddings.bin -cbow 0 -size 50 -window 5 -negative 0 -hs 1 -threads 24 -binary 0 -min-count 0

wikivp:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/wiki_vp_train_lemma.txt -output modelsNdata/wiki_vp_before_50.bin -cbow 0 -size 100 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

common:
	make
	./$(BIN)word2vec -train modelsNdata/coco_train_minus_cs_test.txt -output modelsNdata/word2vec_output_bestmodel_single.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt

# New vqa data
newvqa:
	make
	./$(BIN)word2vec -train modelsNdata/coco_vqa_train_captions_lemma.txt -output modelsNdata/word2vec_vqa_train_captions.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/vqa_train_captions_vocab.txt

genome:
	make
	./$(BIN)word2vec -train modelsNdata/vis-genome/coco_train_captions_genome.txt -output modelsNdata/vis-genome/word2vec_genome_00.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 48 -binary 0 -min-count 0 -load-vocab modelsNdata/vis-genome/genome_train_vocab.txt
	

clean:
	rm -rf word2vec visualFeatures *.o $(BIN)*
