# include "refineFunctions.h"
int noClusters = 0;

// Initializing the refining
void initRefining(){
    long long a, b;
    unsigned long long next_random = 1;

    // Check if noClusters and layer1_size is not 0
    if(layer1_size == 0 || noClusters == 0){
        printf("\nNumber of cluster (%d) | layer1_size (%d) is zero!\n", noClusters, layer1_size);
        exit(1);
    }

    // Setup the network 
    a = posix_memalign((void **)&syn1, 128, (long long)noClusters * layer1_size * sizeof(float));
    if (syn1 == NULL) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    // Initialize the last layer of weights
    for (a = 0; a < noClusters; a++) for (b = 0; b < layer1_size; b++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1[a * layer1_size + b] = (((next_random & 0xFFFF) / (float)65536) - 0.5) / layer1_size;
    }
}

// Evaluate y_i for each output cluster
void computeMultinomial(float* y, int wordId){
    // y stores the multinomial distribution
    float dotProduct = 0, sum = 0;
    long long a, b, offset1, offset2;

    // Offset to access the outer layer weights
    offset1 = wordId * layer1_size;
    for (b = 0; b < noClusters; b++){
        dotProduct = 0;
        // Offset to access the values of hidden layer weights
        offset2 = b * layer1_size;

        for (a = 0; a < layer1_size; a++){
            dotProduct += syn0[offset1 + a] * syn1[offset2 + a];
        }

        // Exponentiating
        y[b] = exp(dotProduct);
    }

    // Normalizing to create a probability measure
    for(b = 0; b < noClusters; b++) sum += y[b];

    if(sum > 0)
        for(b = 0; b < noClusters; b++) y[b] = y[b]/sum;
}

// Updating the weights given the multinomial prediction, word id and true cluster id
void updateWeights(float* y, int wordId, int trueId){
    // compute gradient for outer layer weights, gradient g
    float* e = (float*) malloc(noClusters * sizeof(float));
    long long a, b, c, offset1, offset2;
    float learningRateInner = 0.01, learningRateOuter = 0.01;

    // Computing error
    for(b = 0; b < noClusters; b++){
        if(b == trueId - 1) e[b] = y[b] - 1;
        else e[b] = y[b];
    }
    // Save inner layer weights for correct updates
    float* syn0copy = (float*) malloc(sizeof(float) * layer1_size);

    offset1 = layer1_size * wordId;
    for (c = 0; c < layer1_size; c++) syn0copy[c] = syn0[offset1 + c];
    // compute gradient for inner layer weights
    // update inner layer weights
    // Offset for accessing inner weights
    for(b = 0; b < noClusters; b++){
        // Offset for accesing outer weights
        offset2 = layer1_size * b;
        
        for(c = 0; c < layer1_size; c++)
            syn0[offset1 + c] -= learningRateInner * e[b] * syn1[offset2 + c];
    }

    // compute gradient for outer layer weights
    // update outer layer weights
    for(a = 0; a < noClusters; a++){
        offset2 = layer1_size * a;
        for(b = 0; b < layer1_size; b++){
            syn1[offset2 + b] -= learningRateOuter * e[a] * syn0copy[b];
        }
    }

    // Cleaning the copy
    free(e);
    free(syn0copy);
}

// Computes the multinomial distribution for a phrase
void computeMultinomialPhrase(float* y, int* wordId, int noWords){
    // y stores the multinomial distribution
    float dotProduct = 0, sum = 0;
    long long a, b, i, offset1, offset2;

    // Offset to access the outer layer weights
    for (b = 0; b < noClusters; b++){
        dotProduct = 0;
        // Offset to access the values of hidden layer weights
        offset2 = b * layer1_size;

        // Take mean for all the words
        for(i = 0; i < noWords; i++){
            offset1 = wordId[i] * layer1_size;
            for (a = 0; a < layer1_size; a++)
                dotProduct += 1.0/noWords * syn0[offset1 + a] * syn1[offset2 + a];
        }

        y[b] = exp(dotProduct);
    }

    // Normalizing to create a probability measure
    for(b = 0; b < noClusters; b++) sum += y[b];

    if(sum > 0)
        for(b = 0; b < noClusters; b++) y[b] = y[b]/sum;
}

// Updates the weights for a phrase
void updateWeightsPhrase(float* y, int* wordId, int noWords, int trueId){
    // compute gradient for outer layer weights, gradient g
    float* e = (float*) malloc(noClusters * sizeof(float));
    long long a, b, c, i, offset1, offset2;
    float learningRateOuter = 0.01, learningRateInner = 0.005;

    // Computing error
    for(b = 0; b < noClusters; b++){
        if(b == trueId - 1) e[b] = y[b] - 1;
        else e[b] = y[b];
    }

    // Save inner layer weights for correct updates
    float* syn0copy = (float*) malloc(sizeof(float) * layer1_size * noWords);

    for (i = 0; i < noWords; i++){
        offset1 = layer1_size * wordId[i];
        offset2 = layer1_size * i;
        for (c = 0; c < layer1_size; c++) syn0copy[offset2 + c] = syn0[offset1 + c];
    }

    // compute gradient for inner layer weights
    // update inner layer weights
    // Offset for accessing inner weights
    for(i = 0; i < noWords; i++){
        offset1 = layer1_size * wordId[i];
        for(b = 0; b < noClusters; b++){
            // Offset for accesing outer weights
            offset2 = layer1_size * b;
            
            for(c = 0; c < layer1_size; c++)
                syn0[offset1 + c] -= 1.0/noWords * learningRateInner * e[b] * syn1[offset2 + c];
        }
    }

    // compute gradient for outer layer weights
    // update outer layer weights
    for(i = 0; i < noWords; i++){
        offset1 = layer1_size * i;
        for(a = 0; a < noClusters; a++){
            offset2 = layer1_size * a;
            for(b = 0; b < layer1_size; b++){
                syn1[offset2 + b] -= 1.0/noWords * learningRateOuter * e[a] * syn0copy[offset1 + b];
            }
        }
    }

    free(syn0copy);
    free(e);
}

// Compute the sentence embeddings
// Mean of the embeddings of all the words that are present in the vocab
void computeSentenceEmbeddings(struct Sentence* collection, long noSents){
    printf("\nComputing the sentence embeddings!\n");
    float* mean = (float*) calloc(layer1_size, sizeof(float));
    long i, w, d, offset;

    for( i = 0; i < noSents; i++){
        // reset the mean to zero
        memset(mean, 0, layer1_size);

        // For each sentence, loop over word        
        for( w = 0; w < collection[i].count; w++){
            if(collection[i].index[w] == -1) continue;

            offset = layer1_size * collection[i].index[w];
            // Compute the mean for each dimension
            for (d = 0; d < layer1_size; d++)
                mean[d] += syn0[offset + d];
        }

        // Normalize the mean, if count > 0
        if(collection[i].actCount > 0)
            for (d = 0; d < layer1_size; d++)
                mean[d] /= sqrt(collection[i].actCount);
        
        // If not allocated, allocate memory to embed
        if(collection[i].embed == NULL)
            collection[i].embed = (float*) malloc(sizeof(float) * layer1_size);
        
        // Store the mean 
        memcpy(collection[i].embed, mean, layer1_size * sizeof(float));
    }

    free(mean);
}

// Refine the network based on the cluster id, given sentences
void refineNetworkSentences(struct Sentence* trainSents, long noTrain, enum TrainMode mode){
    long c, i, s, w;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    int wordCount = 0;
    // The starting and ending index for the current sentence in a description
    int startInd, endInd; 

    // Checking if training examples are present
    if(noTrain == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training sentences
    for(i = 0; i < noTrain; i++){
        //printf("Training %ld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(trainSents[i].cId < 1 || trainSents[i].cId > noClusters) {
            printf("\nCluster id (%d) for %ld instance invalid!\n", trainSents[i].cId, i);
            exit(1);
        }

        // Now collecting words for training
        wordCount = 0;
        
        switch(mode){
            // Training each word separately
            case WORDS:
                for(c = 0; c < trainSents[i].count; c++){
                    // Predict the cluster
                    computeMultinomialPhrase(y, trainSents[i].index + c, 1);
                    // Propage the error the embeddings
                    updateWeightsPhrase(y, trainSents[i].index + c, 1, trainSents[i].cId);
                }
                break;

            // Train each sentence separately
            case SENTENCES:
                for(s = 0; s < trainSents[i].sentCount; s++){
                    if(s == 0) startInd = 0;
                    else startInd = trainSents[i].endIndex[s-1] + 1;
                    endInd = trainSents[i].endIndex[s]; 
               
                    //printf("Start, end, number: %d %d %d\n", startInd, endInd, endInd - startInd + 1);
                    // Predict the cluster
                    computeMultinomialPhrase(y, trainSents[i].index + startInd, endInd - startInd + 1);
                    // Propage the error the embeddings
                    updateWeightsPhrase(y, trainSents[i].index + startInd, endInd - startInd + 1, trainSents[i].cId);
                }
                break;

            // Train for each window per sentence (pre-defined window size)
            case WINDOWS:
                for(s = 0; s < trainSents[i].sentCount; s++){
                    if(s == 0) startInd = 0;
                    else startInd = trainSents[i].endIndex[s-1] + 1;
                    endInd = trainSents[i].endIndex[s]; 
               
                    //printf("Start, end, number: %d %d %d\n", startInd, endInd, endInd - startInd + 1);
                    for (w = startInd + windowVP - 1; w <= endInd; w++){
                        //printf("Window: (start, end) : (%d, %d) (%d, %d)\n", w - windowVP + 1, w, startInd, endInd);
                        // Predict the cluster
                        computeMultinomialPhrase(y, 
                                    trainSents[i].index + (w-windowVP+1), windowVP);
                        // Propage the error the embeddings
                        updateWeightsPhrase(y, 
                                trainSents[i].index + (w-windowVP+1), windowVP, trainSents[i].cId);
                    }
                }
                break;

            case DESCRIPTIONS:
                // Predict the cluster
                computeMultinomialPhrase(y, trainSents[i].index, trainSents[i].count);
                // Propage the error the embeddings
                updateWeightsPhrase(y, trainSents[i].index, trainSents[i].count, trainSents[i].cId);
                break;

            default:
                printf("Error in train mode!\n");
                exit(1);
        }
    }
}
