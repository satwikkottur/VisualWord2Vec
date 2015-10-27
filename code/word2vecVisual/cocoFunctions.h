// Header for running COCO scripts
// Just a wrapper for existing functions, got together to make 
// MS COCO training work

# ifndef COCO
# define COCO

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

// Reading the  training sentences
void readTrainSentencesCOCO(char*);
// Reading the cluster ids
void readClusterIdCOCO(char*);
// Tokenize the sentences for training
void tokenizeTrainSentencesCOCO();
// Perform refining with MS COCO
void refineNetworkCOCO();

#endif
