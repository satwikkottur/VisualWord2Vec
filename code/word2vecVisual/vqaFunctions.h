// Header for running VQA scripts
// Just a wrapper for existing functions, got together to make 
// MS VQA training work

# ifndef VQA
# define VQA

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

// External variables
extern enum TrainMode trainMode;
extern int noClusters;
extern int visualFeatSize;

// Reading the  training sentences
void readTrainSentencesVQA(char*);
// Reading the cluster ids
void readClusterIdVQA(char*);
// Tokenize the sentences for training
void tokenizeTrainSentencesVQA();
// Perform refining with MS VQA
void refineNetworkVQA();

// Reading the visual feature file for VQA
void readVisualFeatureFileVQA(char*);

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeaturesVQA(int, char*);

#endif
