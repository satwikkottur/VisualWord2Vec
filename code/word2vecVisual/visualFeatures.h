#ifndef VISUAL_FEATURES
#define VISUAL_FEATURES

// Macros for the visual features
#define MAX_STRING_LENGTH 100
#define NUM_TRAINING 4260
#define NUM_CLUSTERS 10

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
    float* feat;
    // Cluster id assigned to the current instance
    int cId; 
    // Word embedding for the instance
    float* embed; 
};

// Storing the triplets
struct prsTuple prs[NUM_TRAINING];

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
// Storing the feature hash
struct featureWord* featHashWords;
int* featHashInd;
const int featHashSize = 100000;
int featVocabSize = 0;
int featVocabMaxSize = 2000;

// Adding a feature word to the hash
int addFeatureWord(char*);
// Searching for a feature word
int searchFeatureWord(char*);
// Hash for the feature words
int getFeatureWordHash(char*);

#endif
