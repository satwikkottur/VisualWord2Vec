#ifndef DEBUG_FUNCTIONS
#define DEBUG_FUNCTIONS

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
# include "structs.h"
# include "macros.h"

// These contain functions that are used for debugging
// They do not matter once the system is in place
// Hence, written as a separate file

extern struct featureWord* featHashWords;
extern struct prsTuple* train;
extern long noTrain;
extern int noCluster;
extern int visualFeatSize;

void debugVisualFeatureRead(char*);
void debugPRSFeatureRead(char*);

#endif
