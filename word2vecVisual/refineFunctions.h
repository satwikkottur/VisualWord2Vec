// Functions related to generic refining technique
# ifndef REFINE_FUNCS
# define REFINE_FUNCS

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

/*******************************************/
// Functions, variables from orig word2vec
// Declaring the extern variables allowing separation of code
extern long long vocab_size, layer1_size;
extern float *syn0, *syn1, *expTable, *syn0raw;
extern long long vocab_size, layer1_size;
extern int windowVP;
extern int num_threads;

/*******************************************/
// Initializing the refining
void initRefining();
// Initializing the refining, while regression
void initRefiningRegress();

// Evaluate y_i for each output cluster
void computeMultinomial(float*, int);

// Updating the weights given the multinomial prediction, word id and true cluster id
void updateWeights(float*, int, int);

// Computes the multinomial distribution for a phrase
void computeMultinomialPhrase(float*, int*, int);

// Updates the weights for a phrase
void updateWeightsPhrase(float*, int*, int, int);

// Compute the output for regression
void computeOutputRegress(float* y, int);

// Update the weights for regression
void updateWeightsRegress(float*, int, float*);

// Compute the sentence embeddings
// Mean of the embeddings of all the words that are present in the vocab
void computeSentenceEmbeddings(struct Sentence*, long);

// Refine the network based on the cluster id, given sentences
void refineNetworkSentences(struct Sentence*, long, enum TrainMode);
// Thread to do the actual refining
void* refineNetworkThread(void*);
# endif
