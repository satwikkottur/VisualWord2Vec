# include "retriever.h"

// Variables for the current task
static struct Sentence* trainSents;
static struct Sentence* valSents;
static struct Sentence* testSents;

static long noTrain = 0;
static long noVal = 0;
static long noTest = 0;

// Reading the  training sentences
void readTestValRetriever(char* trainPath, char* valPath, char* mapPath, char* testPath, char* testMapPath){
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

    // Use readSentences to read the train sentences
    testSents = *readSentences(testPath, &noSents);
    noTest = noSents;
    // tokenize
    tokenizeSentences(testSents, noTest);

    // Now reading the ground truth for val sentences
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
    
    // Reading the ground truth for test sentences
    mapPtr = fopen(testMapPath, "rb");
    if(mapPtr == NULL){
        printf("File doesnt exist at %s!\n", testMapPath);
        exit(1);
    }
    for(i = 0; i < noTest; i++)
        if(fscanf(mapPtr, "%d\n", &mapId) != EOF)
            testSents[i].gt = mapId;
    fclose(mapPtr);
}

void performRetrieval(){
    // Compute the embeddings
    computeSentenceEmbeddings(valSents, noVal);
    computeSentenceEmbeddings(testSents, noTest);
    computeSentenceEmbeddings(trainSents, noTrain);
    
    // Perform the retrieval for validation set
    performSetRetrieval(valSents, noVal);
    // Perform the retrieval for test set
    performSetRetrieval(testSents, noTest);
}

// Perform retrieval on each of validation and test datasets
void performSetRetrieval(struct Sentence* holder, long noSents){
    // Setup the ranks
    long* ranks = (long*) malloc(sizeof(long) * noSents);

    // Initialize the threads and memory units
    pthread_t* threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    struct RetrieveParameter* params = (struct RetrieveParameter*) 
                        malloc(num_threads * sizeof(struct RetrieveParameter));

    long startId = 0, i;
    long endId = noSents/num_threads;
    for(i = 0; i < num_threads; i++){
        // create the corresponding datastructures
        params[i].startIndex = startId;
        params[i].endIndex = endId;
        params[i].threadId = i;
        params[i].noSents = noSents;
        params[i].sents = holder;
        params[i].ranks = ranks;
    
        // start the threads
        if(pthread_create(&threads[i], NULL, performRetrievalThread, &params[i])){
            fprintf(stderr, "error creating thread\n");
            return;
        }
        
        // compute the start and ends for the next thread
        startId = endId; // start from the next one
        if (i != num_threads - 2)
            // add another chunk if not calculating for the last thread
            endId = endId + noSents/num_threads;
        else
            // everything till the end for the last thread
            endId = noSents;
    }
    
    // wait for all the threads to finish
    for(i = 0 ; i < num_threads; i++)
        if(pthread_join(threads[i], NULL)){
            fprintf(stderr, "error joining thread\n");
            return;
        }

    // Get the recalls
    long r1 = 0, r5 = 0, r10 = 0;
    for(i = 0; i < noSents; i++){
        if (ranks[i] <= 1) r1++;
        if (ranks[i] <= 5) r5++;
        if (ranks[i] <= 10) r10++;
    }
    //printf("%ld %ld %ld\n", r1, r5, r10);

    printf("Recall (%ld) : %f %f %f\n", noSents, ((float)r1)/noSents, 
                                        ((float)r5)/noSents, ((float)r10)/noSents);

    free(params); free(threads); free(ranks);
}

// Perform retrieval thread
void* performRetrievalThread(void* retParams){
    float* scores = (float*) malloc(sizeof(float) * noTrain);
    float dotProduct = 0, magProd = 0;
    struct RetrieveParameter* params = retParams;
    struct Sentence* holder = params->sents;

    long i, j, d, gtRank = 0;
    for (i = params->startIndex; i < params->endIndex; i++){
        if (i % 200 == 0){
            printf("Current instance (%d) : %ld (%ld) ...\n", params->threadId,
                                                i, params->endIndex);
            fflush(stdout);
        }

        for (j = 0; j < noTrain; j++){
            // Compute the dot product for the current train and validation sentence
            dotProduct = 0.0;
            for (d = 0; d < visualFeatSize; d++)
                dotProduct += trainSents[j].embed[d] * holder[i].embed[d];

            magProd = trainSents[j].magnitude * holder[i].magnitude;
            if (magProd)
                scores[j] = dotProduct / magProd;
            else
                scores[j] = 0;
        }
        
        // Pick the score of ground truth and check how many are less (to get rank)
        gtRank = 0;
        for (j = 0; j < noTrain; j++)
            if (scores[holder[i].gt] < scores[j]) gtRank++;

        params->ranks[i] = gtRank + 1;
    }

    free(scores);
    return NULL;
}

