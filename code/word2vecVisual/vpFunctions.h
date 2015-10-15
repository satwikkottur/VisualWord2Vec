// Defines the functions related to the visual paraphrasing task
# ifndef VP_FUNCTIONS
# define VP_FUNCTIONS

// Standard libraries
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <pthread.h>
# include <ctype.h>

# include "structs.h"
# include "macros.h"
# include "helperFunctions.h"
# include "refineFunctions.h"
//*********************************************//
//// Functions from original word2vec
int SearchVocab(char* word);
// Variables from refineFunctions.h
extern int noClusters;
extern long long layer1_size;
extern float* syn0;

//*********************************************//
// Genetic sentence related functions
// Reading sentences
struct Sentence** readSentence(char*, long*);

// Writing sentence embeddings
void writeSentenceEmbeddings(char*, struct Sentence*, long);

// Computing the sentence embeddings
void computeSentenceEmbeddings(struct Sentence*, long);

// Tokenizing sentences
void tokenizeSentences(struct Sentence*, long);

//*********************************************//
// Get the features for all the sentences
//*********************************************//
// Specific to training sentences
// Reading the training sentences
void readVPTrainSentences(char*);

// Reading the visual features
void readVPVisualFeatures(char*);

// Clustering the visual features for the task
void clusterVPVisualFeatures(int, char*);

// Tokenizing training sentences
void tokenizeTrainSentences();

// Save word2vec embeddings for all the VP sentences
void writeVPSentenceEmbeddings();
//*********************************************//
// Refine the network for the VP task
void refineNetworkVP();
#endif
