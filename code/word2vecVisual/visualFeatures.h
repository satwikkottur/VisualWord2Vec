#ifndef VISUAL_FEATURES
#define VISUAL_FEATURES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 50

// Structure to hold information about P,R,S triplets
struct prsTuple{
    char* p;
    char* r;
    char* s;

    float* feature;
};

// Storing the triplets
struct prsTuple prs[4260];

// Reading the feature file
void readFeatureFile(char*);

#endif
