// Header for running text retrieval task
// Just a wrapper for existing functions, got together to make 

# ifndef RETRIEVER
# define RETRIEVER

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
extern int visualFeatSize;
extern int num_threads;

// Reading the  training sentences
void readTestValRetriever(char*, char*, char*);
// Perform retrieval
void performRetrieval();
// Perform retrieval thread
void* performRetrievalThread(void*);
#endif
