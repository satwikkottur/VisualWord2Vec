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

NAMES = visword2vec.o	 \
	visualFeatures.o \
	vpFunctions.o	\
	helperFunctions.o\
	refineFunctions.o\
	liblinearWrapper.o

OBJ=$(NAMES:%=$(BIN)%)
SRCS=$(NAMES:%.o=%.c)

all: $(OBJ)
	$(CC) $(OBJ) -o $(BIN)visword2vec $(BLAS) $(LIBLINEAR) $(CFLAGS)

$(BIN)%.o : %.c
	$(CC) -c $< -o $@ $(CFLAGS)

cs:
	make
	./$(BIN)visword2vec -multi 1 -cs 1 -embed-path data/cs/word2vec_cs.bin -output cs_refined.bin -size 200 -clusters 25 -threads 4
	#./$(BIN)visword2vec -embed-path data/cs/word2vec_cs.bin -output cs_refined.bin -size 200 -clusters 25

vp:
	make
	./$(BIN)visword2vec -vp 1 -embed-path data/vp/word2vec_vp.bin -output vp_refined.bin -size 200 -threads 4 -clusters 100

clean:
	rm -rf visword2vec visualFeatures *.o $(BIN)*
