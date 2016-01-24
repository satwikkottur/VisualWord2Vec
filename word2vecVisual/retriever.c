# include "retriever.h"

// Variables for the current task
static struct Sentence* trainSents;
static struct Sentence* valSents;
static int* featClusterId = NULL;
static long noTrain = 0;
static long noVal = 0;
static long noFeats = 0;
static long* ranks = NULL;

// Reading the  training sentences
void readTestValRetriever(char* trainPath, char* valPath, char* mapPath){
    long noSents = 0;
    // Use readSentences to read the train sentences
    trainSents = *readSentences(trainPath, &noSents);
    noTrain = noSents;
    // tokenize
    tokenizeSentences(trainSents, noTrain);
    
    // readSentences for validation sentences
    valSents = *readSentences(valPath, &noSents);
    noVal = noSents;
    // tokenize
    tokenizeSentences(valSents, noVal);

    // Now reading the maps
    FILE* mapPtr = fopen(mapPath, "rb");

    if(mapPtr == NULL){
        printf("File doesnt exist at %s!\n", mapPath);
        exit(1);
    }

    int i, mapId;
    for(i = 0; i < noVal; i++)
        if(fscanf(mapPtr, "%d\n", &mapId) != EOF)
            valSents[i].gt = mapId;
    fclose(mapPtr);
}

// Perform retrieval
void performRetrieval(){
    if (ranks == NULL) ranks = (long*) malloc(sizeof(long) * noVal);

    // Compute the embeddings for the sentences
    computeSentenceEmbeddings(trainSents, noTrain);
    computeSentenceEmbeddings(valSents, noVal);

    // Initialize the threads and memory units
    pthread_t* threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    struct RefineParameter* params = (struct RefineParameter*) 
                        malloc(num_threads * sizeof(struct RefineParameter));

    long startId = 0, i;
    long endId = noVal/num_threads;
    for(i = 0; i < num_threads; i++){
        // create the corresponding datastructures
        params[i].startIndex = startId;
        params[i].endIndex = endId;
        params[i].threadId = i;
    
        // start the threads
        if(pthread_create(&threads[i], NULL, performRetrievalThread, &params[i])){
            fprintf(stderr, "error creating thread\n");
            return;
        }
        
        // compute the start and ends for the next thread
        startId = endId; // start from the next one
        if (i != num_threads - 2)
            // add another chunk if not calculating for the last thread
            endId = endId + noVal/num_threads;
        else
            // everything till the end for the last thread
            endId = noVal;
    }
    
    // wait for all the threads to finish
    for(i = 0 ; i < num_threads; i++)
        if(pthread_join(threads[i], NULL)){
            fprintf(stderr, "error joining thread\n");
            return;
        }

    // Get the recalls
    long r1 = 0, r5 = 0, r10 = 0;
    for(i = 0; i < noVal; i++){
        if (ranks[i] <= 1) r1++;
        if (ranks[i] <= 5) r5++;
        if (ranks[i] <= 10) r10++;
    }
    //printf("%ld %ld %ld\n", r1, r5, r10);

    printf("Recall : %f %f %f\n", ((float)r1)/noVal, ((float)r5)/noVal, ((float)r10)/noVal);

    free(params); free(threads);
}

// Perform retrieval thread
void* performRetrievalThread(void* retParams){
    float* scores = (float*) malloc(sizeof(float) * noTrain);
    float dotProduct = 0, magProd = 0;
    struct RefineParameter* params = retParams;
    long i, j, d, gtRank = 0;
    for (i = params->startIndex; i < params->endIndex; i++){
        if (i%200 == 0) 
            printf("Current instance (%d) : %ld (%ld) ...\n", params->threadId,
                                                i, params->endIndex);

        for (j = 0; j < noTrain; j++){
            // Compute the dot product for the current train and validation sentence
            dotProduct = 0;
            for (d = 0; d < visualFeatSize; d++)
                dotProduct += trainSents[j].embed[d] * valSents[i].embed[d];

            magProd = trainSents[j].magnitude * valSents[i].magnitude;
            if (trainSents[j].magnitude && valSents[i].magnitude)
                scores[j] = dotProduct / magProd;
            else
                scores[j];

            //printf("%ld : %f %f\n", j, dotProduct, magProd);
        }
        
        // Pick the score of ground truth and check how many are less (to get rank)
        gtRank = 0;
        for (j = 0; j < noTrain; j++)
            if (scores[valSents[i].gt] < scores[j]) gtRank++;

        ranks[i] = gtRank + 1;
    }

    free(scores);
    return NULL;
}

