#ifndef VISUAL_FEATURES
#define VISUAL_FEATURES

// Standard libraries
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <pthread.h>
# include <ctype.h>

// For clustering
# include <yael/vector.h>
# include <yael/kmeans.h>
# include <yael/machinedeps.h>
# include "structs.h"
# include "macros.h"

// Declaring the extern variables allowing separation of code
extern long long vocab_size, layer1_size;
extern float *syn0, *syn1, *expTable;

/************************************************************************/
// Signatures of original functions
/************************************************************************/
typedef float real; // Re-naming float as real
int SearchVocab(char* word);
/************************************************************************/

// getting the vocab indices
struct featureWord constructFeatureWord(char*);
/********************************************************/
// Initializing the feature hash
void initFeatureHash();

// Initializing the refining of network
void initRefining();
void initMultiRefining(); // Training multiple models

// Reading the feature file
void readFeatureFile(char*);

// Reading the cluster id file
void readClusterIdFile(char*);

// Multi character spliting
char* multi_tok(char*, char*);

// Function to refine the network through clusters
void refineNetwork();
// Function to refine the network through clusters, through phrases
void refineNetworkPhrase();
// Function to refine the network through clusters, using multiple models

// Evaluate y_i for each output cluster
void computeMultinomial(float*, int);
// Evaluate y_i for all the phrases
void computeMultinomialPhrase(float*, int*, int);
    
// Updating the weights 
void updateWeights(float*, int, int);
// Updating the weights for phrases
void updateWeightsPhrase(float*, int*, int, int);
// save the embeddings
void saveEmbeddings(char*);

// Compute the embeddings for all the words
void computeEmbeddings();

// save a single feature
void saveFeatureEmbedding(struct featureWord, FILE*);
// Save the vocab for the feature word
void saveFeatureWordVocab(char*);
// Computing the embedding for the feature word 
void computeFeatureEmbedding(struct featureWord*);
/*****************************************/
// Adding a feature word to the hash
int addFeatureWord(char*);
// Searching for a feature word
int searchFeatureWord(char*);
// Hash for the feature words
int getFeatureWordHash(char*);
/*****************************************/
// Functions for kmeans clustering
void readVisualFeatureFile(char*);

// Wrapper for kmeans
void clusterVisualFeatures(int);
/*****************************************/
// Common sense task
int performCommonSenseTask();
// Reading the test and validation files
void readTestValFiles(char*, char*);
// Computing the cos distances
void evaluateCosDistance();
// Computing the test and val scores
void computeTestValScores(struct prsTuple*, long, float, float*);
// Computing the mean AP and basic precision
float* computeMAP(float*, struct prsTuple*, long);
#endif
