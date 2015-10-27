# include "cocoFunctions.h"

// Variables for the current task
static struct Sentence* trainSents;
static long noTrain = 0;

// Reading the  training sentences
void readTrainSentencesCOCO(char* trainPath){
    long noSents = 0;
    // Use readSentences
    trainSents = *readSentences(trainPath, &noSents);

    if(noTrain != 0){
        if(noTrain != noSents){
            printf("Mismatch with number of training examples: %s\n", trainPath);
            exit(1);
        }
    }
    else noTrain = noSents;
    printf("\nRead %ld sentences for training!\n", noTrain);
}

// Reading the cluster ids
void readClusterIdCOCO(char* clusterPath){
    FILE* filePt = fopen(clusterPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", clusterPath);
        exit(1);
    }

    // Keep track of max clsuter id
    int i = 0, clusterId, maxClusterId = 0;
    while(fscanf(filePt, "%d\n", &clusterId) != EOF){
        trainSents[i].cId = clusterId + 1;
        i++;
        if(maxClusterId < clusterId) maxClusterId = clusterId;
    }

    // Sanity check
    if(i != noTrain){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }
    else{
        printf("\nRead cluster file with K = %d\n", maxClusterId+1);
        noClusters = maxClusterId + 1;
    }

    fclose(filePt);
}

// Tokenize the sentences for training
void tokenizeTrainSentencesCOCO(){
    // Call the function to tokenize sentences
    tokenizeSentences(trainSents, noTrain);
}

// Refining the network using COCO
void refineNetworkCOCO(){
    printf("\nRefining using MSCOCO training sentences....\n");
    refineNetworkSentences(trainSents, noTrain, trainMode);
}
