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
# include "liblinearWrapper.h" 
# include "filepaths.h"
//*********************************************//
//// Functions from original word2vec
int SearchVocab(char* word);
// Variables from refineFunctions.h
extern int noClusters;
extern long long layer1_size;
extern float* syn0;
extern int debugModeVP;
extern int windowVP;
extern enum TrainMode trainMode;

//*********************************************//
//*********************************************//
//*********************************************//
// Get the features for all the sentences
//*********************************************//
// Specific to training sentences
// Reading the training sentences
void readVPTrainSentences(char*);

// Reading all sentences along with features
void readVPSentences();
void readVPSentenceFeatures();

// Reading the visual features (abstract)
void readVPAbstractVisualFeatures(char*);

// Clustering the visual features for the task (abstract)
void clusterVPAbstractVisualFeatures(int, char*);

// Tokenizing training sentences
void tokenizeTrainSentences();

// Computing the features for the sentences
void computeSentenceFeatures();

// Save word2vec embeddings for all the VP sentences
void writeVPSentenceEmbeddings();
//*********************************************//
// Refine the network for the VP task
void refineNetworkVP();

// Perform the VP task
void performVPTask();

#endif
