// Wrapper around the liblinear library
# ifndef LIBLIN
# define LIBLIN

// Standard libraries
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <pthread.h>
# include <ctype.h>
# include <limits.h>

// For liblinear
# include "/home/satwik/VisualWord2Vec/libs/liblinear-2.1/linear.h"
# include "structs.h"
# include "macros.h"

/*****/
// External variables
extern long long layer1_size;
/*****/

// Setup the training, model, param and other variables liblinear expects,
// in its format
void learnClassificationModel(struct SentencePair*, long, int);
// Populate the test and train indices
void populateTestTrainIndices(struct SentencePair*, long);
// Create the problem for the training data
void createProblem(struct Sentence*, long);
void createProblemPair(struct SentencePair*); // Given sentence pairs

// Modifying the problem
void modifyProblem(struct Sentence*, long);
void modifyProblemPair(struct SentencePair*); // Given sentence pair

// Creating the parameters
void createParameter();

// Creating the testing nodes
void createTestNodesPair(struct SentencePair*);

// Modifying the testing nodes
void modifyTestNodesPair(struct SentencePair*);

// Create each training instance
struct feature_node* createNodeListSentence(struct Sentence, int);
struct feature_node* createNodeListFeature(float* feature);

// Computing the accuracy for a given bunch of features
float computeAccuracy(struct model*, struct SentencePair*);
# endif
