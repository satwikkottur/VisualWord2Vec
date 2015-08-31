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

// Macros for the visual features
#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

#define MAX_STRING_LENGTH 100
#define NUM_TRAINING 4260
#define NUM_CLUSTERS 10
#define VISUAL_FEATURE_SIZE 1222

// Declaring the extern variables allowing separation of code
extern long long vocab_size, layer1_size;
extern float *syn0, *syn1, *expTable;

// Structure to hold the index information
struct featureWord{
    char* str;
    int count;
    int* index;
};

// Structure to hold information about P,R,S triplets
struct prsTuple{
    int p, r, s;
    //struct featureWord p, r, s;
    
    // visual features for the instance 
    int* feat;
    // Cluster id assigned to the current instance
    int cId; 
    // Word embedding for the instance
    float* embed; 
};

/************************************************************************/
// Signatures of original functions
/************************************************************************/
typedef float real; // Re-naming float as real
int SearchVocab(char* word);

/************************************************************************/
// Storing the triplets
struct prsTuple prs[NUM_TRAINING];
struct prsTuple* test;
struct prsTuple* val;

// getting the vocab indices
struct featureWord constructFeatureWord(char*);
/********************************************************/

// Initializing the refining of network
void initRefining();

// Reading the feature file
void readFeatureFile(char*);

// Reading the cluster id file
void readClusterIdFile(char*);

// Multi character spliting
char* multi_tok(char*, char*);

// Function to refine the network through clusters
void refineNetwork();

// Evaluate y_i for each output cluster
void computeMultinomial(float*, int);
    
// Updating the weights 
void updateWeights(float*, int, int);

// save the embeddings
void saveEmbeddings(char*);

// save a single feature
void saveFeatureEmbedding(struct featureWord, FILE*);
// Save the vocab for the feature word
void saveFeatureWordVocab(char*);
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
void performCommonSenseTask();
// Reading the test and valudation files
void readTestValFiles(char*, struct prsTuple*);
#endif
