# include "debugFunctions.h"

void debugVisualFeatureRead(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    // Iterators
    int i, j;
    for(i = 0; i < noTrain; i++){
        for(j = 0; j < VISUAL_FEATURE_SIZE-1; j++)
            fprintf(filePt, "%f ", train[i].feat[j]);
       
        fprintf(filePt, "%f\n", train[i].feat[VISUAL_FEATURE_SIZE-1]);
    }

    fclose(filePt);
}

void debugPRSFeatureRead(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    // Debugging by saving the PRS back into file and manually checking
    long i;
    for(i = 0; i < noTrain; i++){
        fprintf(filePt, "<%s:%s:%s>\n", featHashWords[train[i].p].str, 
                                        featHashWords[train[i].s].str,
                                        featHashWords[train[i].r].str);
    }

    fclose(filePt);
}
