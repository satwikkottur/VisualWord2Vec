# include "debugFunctions.h"

void debugVisualFeatureRead(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    // Iterators
    int i, j;
    for(i = 0; i < noTrain; i++){
        for(j = 0; j < visualFeatSize-1; j++)
            fprintf(filePt, "%f ", trainTuples[i].feat[j]);
       
        fprintf(filePt, "%f\n", trainTuples[i].feat[visualFeatSize-1]);
    }

    fclose(filePt);
}

void debugPRSFeatureRead(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    // Debugging by saving the PRS back into file and manually checking
    long i;
    for(i = 0; i < noTrain; i++){
        fprintf(filePt, "<%s:%s:%s>\n", featHashWords[trainTuples[i].p].str, 
                                        featHashWords[trainTuples[i].s].str,
                                        featHashWords[trainTuples[i].r].str);
    }

    fclose(filePt);
}
