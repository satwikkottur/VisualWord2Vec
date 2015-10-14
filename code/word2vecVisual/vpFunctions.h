// Defines the functions related to the visual paraphrasing task
# ifndef VP_FUNCTIONS
# define VP_FUNCTIONS

// Standard libraries
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <pthread.h>
# include <ctype.h>

# include "structs.h"
# include "macros.h"


// Reading the training sentences
void readVPTrainSentences(char*);

// Reading the visual features
void readVPVisualFeatures(char*);

#endif
