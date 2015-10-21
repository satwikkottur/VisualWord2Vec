# include "refineFunctions.h"
int noClusters = 0;

// Initializing the refining
void initRefining(){
    long long a, b;
    unsigned long long next_random = 1;

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

