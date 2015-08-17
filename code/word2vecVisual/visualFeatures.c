# include "functionSigns.h"
# include "visualFeatures.h"

// reading feature file
void readFeatureFile(char* filePath){
    // Opening the file
    FILE* filePt = fopen(filePath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", filePath);
        exit(1);
    }

    printf("\nReading %s...\n", filePath);

    char pWord[MAX_STRING_LENGTH], sWord[MAX_STRING_LENGTH], rWord[MAX_STRING_LENGTH];
    // Read and store the contents
    int noTuples = 0;
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]>\n", pWord, sWord, rWord) != EOF){
        // Getting the indices for p, s, r
        struct prsTuple newTuple;/* = {.p = {findTupleIndex(pWord)},
                                    .r = {findTupleIndex(rWord)},
                                    .s = {findTupleIndex(sWord)},
                                    .cId = -1,
                                    .feat = NULL,
                                    .embed = NULL};*/
        //printf("%s : %s : %s\n", p, s, r);
        newTuple.p = findTupleIndex(pWord);
        newTuple.s = findTupleIndex(sWord);
        newTuple.r = findTupleIndex(rWord);

        prs[noTuples] = newTuple;
        noTuples++;
    }

    // Sanity check
    if(noTuples != NUM_TRAINING){
        printf("\nNumber of training instances dont match in feature file!\n");
        exit(1);
    }

    fclose(filePt);
    printf("File read with %d tuples\n\n", noTuples);
}

// Reading the cluster ids
void readClusterIdFile(char* clusterPath){
    FILE* filePt = fopen(clusterPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", clusterPath);
        exit(1);
    }

    int i = 0, clusterId;
    while(fscanf(filePt, "%d\n", &clusterId) != EOF){
        if(prs[i].cId == -1) prs[i].cId = clusterId;
        i++;
    }

    // Sanity check
    if(i != NUM_TRAINING){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }

    fclose(filePt);
}

// Finding the indices of words for P,R,S
struct featureWord findTupleIndex(char* word){
    int index = SearchVocab(word); 

    struct featureWord feature = {.str = word};
    
    // Do something if not in vocab
    if(index == -1) {
        printf("Not in vocab -> %s\n", word) ;
    } else{
        //printf("In Vocab -> %s\n", word);
        feature.count = 1;
        feature.index = (int *) malloc(1);
        feature.index[0] = index;
    }

    return feature;
}
