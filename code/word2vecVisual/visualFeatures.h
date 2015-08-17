#ifndef VISUAL_FEATURES
#define VISUAL_FEATURES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 50
#define NUM_TRAINING 4260

// Structure to hold the index information
struct featureWord{
    char* str;
    int count;
    int* index;
};

// Structure to hold information about P,R,S triplets
struct prsTuple{
    struct featureWord p, r, s;

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
struct featureWord findTupleIndex(char*);

// Reading the feature file
void readFeatureFile(char*);

// Reading the cluster id file
void readClusterIdFile(char*);


#endif
