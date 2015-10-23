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

#all: visualFeatures word2vec
#	$(CC) visualFeatures.o word2vec.o -o word2vec $(CFLAGS)
all: project

project: $(BIN)word2vec.o $(BIN)visualFeatures.o $(BIN)debugFunctions.o $(BIN)vpFunctions.o $(BIN)helperFunctions.o $(BIN)refineFunctions.o $(BIN)liblinearWrapper.o $(LIBLINEAR)
	$(CC) $(BIN)word2vec.o $(BIN)visualFeatures.o $(BIN)debugFunctions.o $(BIN)vpFunctions.o $(BIN)helperFunctions.o $(BIN)refineFunctions.o $(BIN)liblinearWrapper.o $(LIBLINEAR) -o $(BIN)word2vec $(CFLAGS)
$(BIN)word2vec.o : word2vec.c
	$(CC) -c word2vec.c -o $(BIN)word2vec.o $(CFLAGS)
$(BIN)visualFeatures.o : visualFeatures.c
	$(CC) -c visualFeatures.c -o $(BIN)visualFeatures.o $(CFLAGS)
$(BIN)debugFunctions.o : debugFunctions.c
	$(CC) -c debugFunctions.c -o $(BIN)debugFunctions.o $(CFLAGS)
$(BIN)vpFunctions.o : vpFunctions.c
	$(CC) -c vpFunctions.c -o $(BIN)vpFunctions.o $(CFLAGS)
$(BIN)helperFunctions.o : helperFunctions.c
	$(CC) -c helperFunctions.c -o $(BIN)helperFunctions.o $(CFLAGS)
$(BIN)refineFunctions.o : refineFunctions.c
	$(CC) -c refineFunctions.c -o $(BIN)refineFunctions.o $(CFLAGS)
$(BIN)liblinearWrapper.o : liblinearWrapper.c
	$(CC) -c liblinearWrapper.c -o $(BIN)liblinearWrapper.o $(BLAS) $(CFLAGS)

run:
	make
	#./$(BIN)word2vec -train modelsNdata/vp/vp_train_sentences_lemma.txt -output word2vec_output_makerun.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0
	#./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/wiki_vp_train_lemma.txt -output word2vec_output_makerun.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0
	#./$(BIN)word2vec -train modelsNdata/vp/coco_lemma_vp_lemma.txt -output modelsNdata/vp/word2vec_coco_vp_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

vp:
	make
	./$(BIN)word2vec -train modelsNdata/vp/coco_lemma_vp_lemma.txt -output modelsNdata/word2vec_vp_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

wiki:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/wiki_vp_train_lemma.txt -output modelsNdata/word2vec_after_refine_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

big:
	make
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/wiki_vp_train_lemma.txt -output modelsNdata/word2vec_after_refine_lemma.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0
	#./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/enwik9 -output modelsNdata/word2vec_after_refine_bestmodel_wiki.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 24 -binary 0 -min-count 0
	#./$(BIN)word2vec -train modelsNdata/coco_train_minus_cs_test.txt -output modelsNdata/word2vec_output_bestmodel_single.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0 -read-vocab modelsNdata/large-vocab.txt
	# Getting the visualword2vec for VP training + MS COCO
	#./$(BIN)word2vec -train modelsNdata/coco_vp_train.txt -output modelsNdata/word2vec_vp_train.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

newline:
	./$(BIN)word2vec -train /home/satwik/VisualWord2Vec/data/coco_train_minus_cs_test_tokenized_stops.txt -output word2vec_output_newline.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -threads 12 -binary 0 -min-count 0

clean:
	rm -rf word2vec visualFeatures *.o $(BIN)*
