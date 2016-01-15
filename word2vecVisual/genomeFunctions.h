// Header for running visual genome dataset
// Just a wrapper for existing functions, got together to make 
// Visual Genome 

# ifndef GENOME
# define GENOME

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
extern int num_threads;

// Reading the  training sentences, along with the map ids
void readTrainSentencesGenome(char*);
// Reading the cluster ids
void readClusterIdGenome(char*);
// Tokenize the sentences for training
void tokenizeTrainSentencesGenome();
// Perform refining with visual genome
void refineNetworkGenome();

// Reading the visual feature file for VQA
void readVisualFeatureFileGenome(char*);

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeaturesGenome(int, char*);

#endif
