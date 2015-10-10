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
extern struct vocab_word* vocab;
extern int permuteMAP;

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
void refineMultiNetwork();
// Function to refine the network for multi models using phrases
void refineMultiNetworkPhrase();

// Evaluate y_i for each output cluster
void computeMultinomial(float*, int);
// Evaluate y_i for all the phrases
void computeMultinomialPhrase(float*, int*, int);
    
// Updating the weights 
void updateWeights(float*, int, int);
// Updating the weights for phrases
void updateWeightsPhrase(float*, int*, int, int);

//------------------------------------------------------------
// Compute the embeddings for all the words
void computeEmbeddings();
// Computing the embeddings in case of multi-model
void computeMultiEmbeddings();

// Computing the embedding for the feature word 
void computeFeatureEmbedding(struct featureWord*);
// Computing the embedding for feature word in case of multi model
void computeMultiFeatureEmbedding(struct featureWord*);
//------------------------------------------------------------
// save the embeddings
void saveEmbeddings(char*);
// save a single feature
void saveFeatureEmbedding(struct featureWord, FILE*);
// save the embeddings for the multi model
void saveMultiEmbeddings(char*);
// save a single feature, for multi model
void saveMultiFeatureEmbedding(struct featureWord, FILE*);

// Save the vocab for the feature word
void saveFeatureWordVocab(char*);
// Save the vocab for the feature word(split into words)
void saveFeatureWordVocabSplit(char*);

// Saving tuple embeddings and tuple words in separate files
void saveTupleEmbeddings(char*, char*, struct prsTuple*, float*, float*, int*, int);
// Saving tuple embeddings and tuple words in separate files, for multimodel
void saveMultiTupleEmbeddings(char*, char*, struct prsTuple*, float*, float*, int*, int);
// Saving the tuples only along with scores
void saveTupleScores(char*, struct prsTuple*,float*, float*, int*, int);
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
void clusterVisualFeatures(int, char*);
/*****************************************/
// Common sense task
int performCommonSenseTask(float*);
// Common sense task with multi models
int performMultiCommonSenseTask(float*);
// Reading the test and validation files
void readTestValFiles(char*, char*);
// Computing the cos distances
void evaluateCosDistance();
// Computing the cos distances for multi-model
void evaluateMultiCosDistance();
// Computing the test and val scores
void computeTestValScores(struct prsTuple*, long, float, float*);
// Computing the test and val scores with multi models
void computeMultiTestValScores(struct prsTuple*, long, float, float*);
// Computing the mean AP and basic precision
float* computeMAP(float*, struct prsTuple*, long);
// Computing the mean AP and basic precision, with a permutation specified
float* computePermuteMAP(float*, struct prsTuple*, int*, long);
// Saving the word2vec vectors for further use
void saveWord2Vec(char*);
// Loading the word2vec vectors for further use
void loadWord2Vec(char*);

/*****************************************/
// Visualization and qualitative analysis
// Get the best test tuples with maximum improvement
void findBestTestTuple(float*, float*);
#endif
