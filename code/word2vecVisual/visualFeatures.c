# include "visualFeatures.h"
// Include the random permutation code
# include "randompermute.h"

// Storing the feature hash (globals)
struct featureWord* featHashWords; // Vocab for feature words
int* featHashInd; // Storing the hash indices that reference to vocab
const int featHashSize = 200000; // Size of the hash for feature words
int featVocabSize = 0; // Actual vocab size for feature word 
int featVocabMaxSize = 5000; // Maximum number of feature vocab
long noTrain = 0, noTest = 0, noVal = 0; // Number of test and validation variables
float* cosDist; // Storing the cosine distances between all the feature vocabulary
float* valScore, *testScore; // Storing the scores for test and val
// Storing the cosine distances between all the feature vocabulary (multimodel)
float *cosDistP, *cosDistR, *cosDistS; 
int verbose = 0; // Printing which function is being executed
int noClusters = 0; // Number of clusters to be used
int visualFeatSize = 0; // Size of the visual features used
float prevValAcc = 0, prevTestAcc = 0;

struct prsTuple *train, *test, *val;
float *syn0P, *syn0S, *syn0R;
float *syn1P, *syn1S, *syn1R;

// Forcing limited number of data
int forceTrainData = 0;
int forcedTrainSize = 4000;
/***************************************************************************/
// reading feature file
void readFeatureFile(char* filePath){
    // Opening the file
    FILE* filePt = fopen(filePath, "rb");

    char pWord[MAX_STRING], sWord[MAX_STRING], rWord[MAX_STRING];
    long noTuples = 0, i;

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", filePath);
        exit(1);
    }

    printf("\nReading %s...\n", filePath);

    // Compute the number of lines / instances and check with current existing variable
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]>\n", pWord, rWord, sWord) != EOF)
        noTuples++;
    // Rewind the stream and read again
    rewind(filePt);
    
    if(forceTrainData)
        noTuples = forcedTrainSize;

    // Initialize and save the feature words
    train = (struct prsTuple*) malloc(sizeof(struct prsTuple) * noTuples);
    // Read and store the contents
    for(i = 0; i < noTuples; i++){
        if(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]>\n", pWord, sWord, rWord) != EOF){
            // Getting the indices for p, s, r
            train[i].p = addFeatureWord(pWord);
            train[i].r = addFeatureWord(rWord);
            train[i].s = addFeatureWord(sWord);
        }
    }

    // Checking with noTrain, if exists, else initializing
    if(noTrain != 0){
        if(noTrain != noTuples){
            printf("Mismatch with number of training examples: %s\n", filePath);
            exit(1);
        }
    }
    else noTrain = noTuples;

    fclose(filePt);
    printf("File read with %ld tuples\n\n", noTuples);
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
        //if(train[i].cId == -1) train[i].cId = clusterId;
        train[i].cId = clusterId;
        i++;
    }

    // Sanity check
    if(i != noTrain){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }

    // Debugging
    /*for(i = 0; i < noTrain; i++){
        printf("%s : %s : %s : (%d)\n", train[i].p.str, train[i].s.str, train[i].r.str, train[i].cId);
    }*/

    fclose(filePt);
}

// Reading the visual feature file
void readVisualFeatureFile(char* fileName){
    FILE* filePt = fopen(fileName, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", fileName);
        exit(1);
    }

    float feature;
    int i, noLines = 0;

    // Read the first line and get the feature size
    fscanf(filePt, "%d", &visualFeatSize);
    printf("Visual features are of size : %d...\n", visualFeatSize);

    // Reading till EOF
    while(fscanf(filePt, "%f", &feature) != EOF){
        train[noLines].feat = (float*) malloc(sizeof(float) * visualFeatSize);
        // Save the already read feature
        train[noLines].feat[0] = feature;

        for(i = 1; i < visualFeatSize; i++){
            //printf("%f ", feature);
            fscanf(filePt, "%f", &feature);
            train[noLines].feat[i] = feature;
        }
        //printf("%f\n", feature);

        //printf("Line : %d\n", noLines);
        noLines++;

        // If forced train size, quit after noLines exceeds the max size
        if(forceTrainData)
            if(noLines == forcedTrainSize)
                break;
    }

    printf("\nRead visual features for %d tuples...\n", noLines);

    // Closing the file
    fclose(filePt);
}

// Finding the indices of words for P,R,S
struct featureWord constructFeatureWord(char* word){
    int index = SearchVocab(word); 

    struct featureWord feature;
    feature.str = (char*) malloc(MAX_STRING);
    strcpy(feature.str, word);

    // Initialize the future embedding
    feature.magnitude = 0;
    feature.embed = NULL;
    feature.embedR = NULL;
    feature.embedS = NULL;
    feature.embedP = NULL;
    
    // Do something if not in vocab
    if(index == -1) {
        //printf("Not in vocab -> %s : %s\n", word, "") ;
        int count=0, i;

        // Split based on 's
        char* token = (char*) malloc(MAX_STRING);
        strcpy(token, word);

        char* first = multi_tok(token, "'s");
        char* second = multi_tok(NULL, "'s");

        // Join both the parts without the 's
        if(second != NULL) token = strcat(first, second);
        else token = first;

        char* temp = (char*) malloc(MAX_STRING);
        strcpy(temp, token);
        
        // Remove ' ', ',', '.', '?', '!', '\', '/'
        char* delim = " .,/!?\\";
        token = strtok(token, delim);
        // Going over the token to determine the number of parts
        while(token != NULL){
            count++;
            token = strtok(NULL, delim);
        }

        // Nsmallow store the word components, looping over them
        feature.index = (int*) malloc(count * sizeof(int));
        feature.count = count;
        
        token = strtok(temp, delim);
        count = 0;
        while(token != NULL){
            // Convert the token into lower case
            for(i = 0; token[i]; i++) token[i] = tolower(token[i]);
           
            // Save the index
            feature.index[count] = SearchVocab(token);
            //if(feature.index[count] == -1)
            //   printf("Word not found in dictionary => %s\t |  %s\n", token, word);

            //printf("%d \t", feature.index[count]);
            token = strtok(NULL, delim);
            count++;
        }
        //printf("\n");

    } else{
        //printf("In Vocab -> %s\n", word);
        feature.count = 1;
        feature.index = (int *) malloc(sizeof(int));
        feature.index[0] = index;
    }

    return feature;
}

// Initializing the feature hash
void initFeatureHash(){
    long a;

    // Setting up the hash
    featHashWords = (struct featureWord *) malloc(sizeof(struct featureWord) * featVocabMaxSize);
    featHashInd = (int*) malloc(sizeof(int) * featHashSize);
    for(a = 0; a < featHashSize; a++)
        featHashInd[a] = -1;
}

// Initializing the refining
void initRefining(){
    long long a, b;
    unsigned long long next_random = 1;

    // Setup the network 
    a = posix_memalign((void **)&syn1, 128, (long long)noClusters * layer1_size * sizeof(real));
    if (syn1 == NULL) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    // Initialize the last layer of weights
    for (a = 0; a < noClusters; a++) for (b = 0; b < layer1_size; b++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}

// Initializing the multi-model refining
void initMultiRefining(){
    long long a, b;
    unsigned long long next_random = 1;

    // Allocate and Make copies of syn0 as syn0P, syn0R, syn0S
    a = posix_memalign((void **)&syn0P, 128, (long long)vocab_size * layer1_size * sizeof(real));
    a = posix_memalign((void **)&syn0S, 128, (long long)vocab_size * layer1_size * sizeof(real));
    a = posix_memalign((void **)&syn0R, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn0P == NULL || syn0R == NULL || syn0S == NULL) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }
    memcpy(syn0P, syn0, (size_t)vocab_size * layer1_size * sizeof(float));
    memcpy(syn0S, syn0, (size_t)vocab_size * layer1_size * sizeof(float));
    memcpy(syn0R, syn0, (size_t)vocab_size * layer1_size * sizeof(float));

    // Initialize syn1P, syn1R, syn1S
    // Setup the network 
    a = posix_memalign((void **)&syn1P, 128, (long long)noClusters * layer1_size * sizeof(real));
    a = posix_memalign((void **)&syn1S, 128, (long long)noClusters * layer1_size * sizeof(real));
    a = posix_memalign((void **)&syn1R, 128, (long long)noClusters * layer1_size * sizeof(real));
    if (syn1P == NULL || syn1R == NULL || syn1S == NULL) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    // Initialize the last layer of weights
    for (a = 0; a < noClusters; a++) for (b = 0; b < layer1_size; b++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1R[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;

        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1S[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;

        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1P[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }
}

// Refine the network through clusters
void refineNetwork(){
    long long c, i;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    struct featureWord p, s, r;

    // Checking if training examples are present
    if(noTrain == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training instance
    for(i = 0; i < noTrain; i++){
        //printf("Training %lld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(train[i].cId < 1 || train[i].cId > noClusters) {
            printf("\nCluster id (%d) for %lld instance invalid!\n", train[i].cId, i);
            exit(1);
        }

        //printf("Counts : %d %d %d\n", train[i].p.count, train[i].s.count, train[i].r.count);

        // Updating the weights for P
        p = featHashWords[train[i].p];
        for(c = 0; c < p.count; c++){
            // If not in vocab, continue
            if(p.index[c] == -1) continue;
            //printf("p: %d %d\n", p.index[c], p.count);

            // Predict the cluster
            computeMultinomial(y, p.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, p.index[c], train[i].cId);
        }
        
        // Updating the weights for S
        s = featHashWords[train[i].s];
        for(c = 0; c < s.count; c++){
            // If not in vocab, continue
            if(s.index[c] == -1) continue;
            //printf("s: %d %d\n", s.index[c], s.count);

            // Predict the cluster
            computeMultinomial(y, s.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, s.index[c], train[i].cId);
        }

        // Updating the weights for R
        r = featHashWords[train[i].r];
        for(c = 0; c < r.count; c++){
            // If not in vocab, continue
            if(r.index[c] == -1) continue;
            //printf("r: %d %d\n", r.index[c], r.count);

            // Predict the cluster
            computeMultinomial(y, r.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, r.index[c], train[i].cId);
        }
    }
}

// Refine the network through clusters using multiple models
void refineMultiNetwork(){
    long long c, i;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    struct featureWord p, s, r;

    // Checking if training examples are present
    if(noTrain == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training instance
    for(i = 0; i < noTrain; i++){
        //if (verbose)
        //    printf("Training %lld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(train[i].cId < 1 || train[i].cId > noClusters) {
            printf("\nCluster id (%d) for %lld instance invalid!\n", train[i].cId, i);
            exit(1);
        }

        //printf("Counts : %d %d %d\n", train[i].p.count, train[i].s.count, train[i].r.count);

        // Updating the weights for P
        p = featHashWords[train[i].p];
        for(c = 0; c < p.count; c++){
            // If not in vocab, continue
            if(p.index[c] == -1) continue;
            //printf("p: %d %d\n", p.index[c], p.count);

            // Point syn0 and syn1 to P model
            syn0 = syn0P;
            syn1 = syn1P;

            // Predict the cluster
            computeMultinomial(y, p.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, p.index[c], train[i].cId);
        }
        
        // Updating the weights for S
        s = featHashWords[train[i].s];
        for(c = 0; c < s.count; c++){
            // If not in vocab, continue
            if(s.index[c] == -1) continue;
            //printf("s: %d %d\n", s.index[c], s.count);

            // Point syn0 and syn1 to S model
            syn0 = syn0S;
            syn1 = syn1S;

            // Predict the cluster
            computeMultinomial(y, s.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, s.index[c], train[i].cId);
        }

        // Updating the weights for R
        r = featHashWords[train[i].r];
        for(c = 0; c < r.count; c++){
            // If not in vocab, continue
            if(r.index[c] == -1) continue;
            //printf("r: %d %d\n", r.index[c], r.count);

            // Point syn0 and syn1 to R model
            syn0 = syn0R;
            syn1 = syn1R;

            // Predict the cluster
            computeMultinomial(y, r.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, r.index[c], train[i].cId);
        }
    }
}

// Refine the network through clusters, for phrases
void refineNetworkPhrase(){
    long long c, i;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    struct featureWord p, s, r;
    int* wordList = (int*) malloc(100 * sizeof(int));
    int wordCount = 0;

    // Checking if training examples are present
    if(noTrain == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training instance
    for(i = 0; i < noTrain; i++){
        //printf("Training %lld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(train[i].cId < 1 || train[i].cId > noClusters) {
            printf("\nCluster id (%d) for %lld instance invalid!\n", train[i].cId, i);
            exit(1);
        }

        // Now collecting words for training
        /*****************************************/
        // Updating the weights for P
        p = featHashWords[train[i].p];
        wordCount = 0;
        
        for(c = 0; c < p.count; c++){
            // If not in vocab, continue
            if(p.index[c] == -1) continue;

            wordList[wordCount] = p.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
        // Updating the weights for S
        s = featHashWords[train[i].s];
        wordCount = 0;
        
        for(c = 0; c < s.count; c++){
            // If not in vocab, continue
            if(s.index[c] == -1) continue;

            wordList[wordCount] = s.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
        // Updating the weights for R
        r = featHashWords[train[i].r];
        wordCount = 0;
        
        for(c = 0; c < r.count; c++){
            // If not in vocab, continue
            if(r.index[c] == -1) continue;

            wordList[wordCount] = r.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
    }
}

// Refine the network through clusters, for phrases. for multiple models
void refineMultiNetworkPhrase(){
    long long c, i;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    struct featureWord p, s, r;
    int* wordList = (int*) malloc(100 * sizeof(int));
    int wordCount = 0;

    // Checking if training examples are present
    if(noTrain == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training instance
    for(i = 0; i < noTrain; i++){
        //printf("Training %lld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(train[i].cId < 1 || train[i].cId > noClusters) {
            printf("\nCluster id (%d) for %lld instance invalid!\n", train[i].cId, i);
            exit(1);
        }

        // Now collecting words for training
        /*****************************************/
        // Updating the weights for P
        p = featHashWords[train[i].p];
        wordCount = 0;
        
        for(c = 0; c < p.count; c++){
            // If not in vocab, continue
            if(p.index[c] == -1) continue;

            wordList[wordCount] = p.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Pointing syn0, syn1 to syn0P,syn1P
        syn0 = syn0P;
        syn1 = syn1P;

        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
        // Updating the weights for S
        s = featHashWords[train[i].s];
        wordCount = 0;
        
        for(c = 0; c < s.count; c++){
            // If not in vocab, continue
            if(s.index[c] == -1) continue;

            wordList[wordCount] = s.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Pointing syn0, syn1 to syn0S,syn1S
        syn0 = syn0S;
        syn1 = syn1S;
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
        // Updating the weights for R
        r = featHashWords[train[i].r];
        wordCount = 0;
        
        for(c = 0; c < r.count; c++){
            // If not in vocab, continue
            if(r.index[c] == -1) continue;

            wordList[wordCount] = r.index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Pointing syn0, syn1 to syn0R,syn1R
        syn0 = syn0R;
        syn1 = syn1R;
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
        /*****************************************/
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

        // Exponential (clip if less or greater than the limit)
        //if (dotProduct <= - MAX_EXP) dotProduct = -MAX_EXP;
        //else if (dotProduct >= MAX_EXP) dotProduct = MAX_EXP;
        //else dotProduct = expTable[(int) ((dotProduct + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
 
        // Exponentiating
        //printf();
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

        // Exponential (clip if less or greater than the limit)
        //if (dotProduct <= - MAX_EXP) dotProduct = -MAX_EXP;
        //else if (dotProduct >= MAX_EXP) dotProduct = MAX_EXP;
        //else dotProduct = expTable[(int) ((dotProduct + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

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
    float learningRateOuter = 0.001, learningRateInner = 0.001;

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

// Saving the feature embeddings needed for comparing, at the given file name
void saveEmbeddings(char* saveName){
    FILE* filePt = fopen(saveName, "wb");
    int i;

    // Re-compute the embeddings before saving
    computeEmbeddings();
    
    // Go through the vocab and save the embeddings
    for(i = 0; i < featVocabSize; i++)
        saveFeatureEmbedding(featHashWords[i], filePt);
    
    fclose(filePt);
}

// Save a particular embedding
void saveFeatureEmbedding(struct featureWord feature, FILE* filePt){
    // Saving to the file
    int i;
    for(i = 0; i < layer1_size - 1; i++)
        fprintf(filePt, "%f ", feature.embed[i]);
    fprintf(filePt, "%f\n", feature.embed[layer1_size-1]);
}

// Saving the feature embeddings needed for comparing, at the given file name, for multi model
void saveMultiEmbeddings(char* saveName){
    FILE* filePt = fopen(saveName, "wb");
    int i;

    // Re-compute the embeddings before saving
    computeMultiEmbeddings();
    
    // Go through the vocab and save the embeddings
    for(i = 0; i < featVocabSize; i++)
        saveMultiFeatureEmbedding(featHashWords[i], filePt);
    
    fclose(filePt);
}

// Save a particular embedding, for multi model
void saveMultiFeatureEmbedding(struct featureWord feature, FILE* filePt){
    // Saving to the file, all the three embeddings
    int i;
    for(i = 0; i < layer1_size - 1; i++)
        fprintf(filePt, "%f ", feature.embedP[i]);
    fprintf(filePt, "%f\n", feature.embedP[layer1_size-1]);

    for(i = 0; i < layer1_size - 1; i++)
        fprintf(filePt, "%f ", feature.embedR[i]);
    fprintf(filePt, "%f\n", feature.embedR[layer1_size-1]);

    for(i = 0; i < layer1_size - 1; i++)
        fprintf(filePt, "%f ", feature.embedS[i]);
    fprintf(filePt, "%f\n", feature.embedS[layer1_size-1]);
}

// Saving the feature vocab
void saveFeatureWordVocab(char* fileName){
    FILE* filePt = fopen(fileName, "wb");
    int i;

    // Go through the vocab and save the embeddings
    for(i = 0; i < featVocabSize; i++)
        fprintf(filePt, "%s\n", featHashWords[i].str);

    fclose(filePt);
}

// Saving tuples
void saveTupleEmbeddings(char* tupleFile, char* embedFile, struct prsTuple* holder, float* baseScores, float* bestScores, int* members, int noMembers){
    // Compute the embeddings before saving
    computeEmbeddings();
    
    // Save the tuples along with scores
    saveTupleScores(tupleFile, holder, baseScores, bestScores, members, noMembers); 
    
    // Save the embeddings for the tuples
    FILE* filePt = fopen(embedFile, "wb");

    int i, j, index;
    for(i = 0; i < noMembers; i++){
        index = members[i];
        // Store P embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].p].embed[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].p].embed[layer1_size-1]);
    
        // Store S embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].r].embed[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].r].embed[layer1_size-1]);

        // Store R embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].s].embed[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].s].embed[layer1_size-1]);
    }

    fclose(filePt);
}

// Saving tuples
void saveMultiTupleEmbeddings(char* tupleFile, char* embedFile, struct prsTuple* holder, float* baseScores, float* bestScores, int* members, int noMembers){
    // Compute the embeddings before saving
    computeMultiEmbeddings();
    
    // Save the tuples along with scores
    saveTupleScores(tupleFile, holder, baseScores, bestScores, members, noMembers); 
    
    // Save the embeddings for the tuples
    FILE* filePt = fopen(embedFile, "wb");

    int i, j, index;
    for(i = 0; i < noMembers; i++){
        index = members[i];
        // Store P embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].p].embedP[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].p].embedP[layer1_size-1]);
    
        // Store S embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].r].embedR[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].r].embedR[layer1_size-1]);

        // Store R embed
        for(j = 0; j < layer1_size-1; j++)
            fprintf(filePt, "%f ", featHashWords[holder[index].s].embedS[j]);
        fprintf(filePt, "%f\n", featHashWords[holder[index].s].embedS[layer1_size-1]);
    }

    fclose(filePt);
}

// Saving only tuples
void saveTupleScores(char* tupleFile, struct prsTuple* holder, float* baseScores, float* bestScores, int* members, int noMembers){
    FILE* filePt = fopen(tupleFile, "wb");
        
    int i, index;
    for (i = 0; i < noMembers; i++){
        index = members[i]; 

        // Save the tuples along with scores
        fprintf(filePt, "<%s:%s:%s>(%f,%f)\n", featHashWords[holder[index].p].str, 
                                                featHashWords[holder[index].r].str,
                                                featHashWords[holder[index].s].str,
                                                baseScores[index],
                                                bestScores[index]);
    }

    // Close the file
    fclose(filePt);
}

// Compute embeddings
void computeEmbeddings(){
    long i;
    // Computing the feature embeddings
    for(i = 0; i < featVocabSize; i++){
        if(featHashWords[i].embed == NULL)
            // Allocate and then compute the feature embedding
            featHashWords[i].embed = (float*) malloc(layer1_size * sizeof(float));

        // Computing the feature embedding
        computeFeatureEmbedding(&featHashWords[i]);
    }
}

// Compute embeddings for multi models
void computeMultiEmbeddings(){
    long i;
    // Computing the feature embeddings
    for(i = 0; i < featVocabSize; i++){
        // Allocate and then compute the feature embedding for each model
        if(featHashWords[i].embedR == NULL)
            featHashWords[i].embedR = (float*) malloc(layer1_size * sizeof(float));
        
        if(featHashWords[i].embedP == NULL)
            featHashWords[i].embedP = (float*) malloc(layer1_size * sizeof(float));

        if(featHashWords[i].embedS == NULL)
            featHashWords[i].embedS = (float*) malloc(layer1_size * sizeof(float));

        // Computing the feature embedding
        computeMultiFeatureEmbedding(&featHashWords[i]);
    }
}

// Compute embedding for a feature word
void computeFeatureEmbedding(struct featureWord* feature){
    // Go through the current feature and get the mean of components
    int i, c, actualCount = 0;
    long long offset;
    float* mean;
    mean = (float*) calloc(layer1_size, sizeof(float));

    // Get the mean feature for the word
    for(c = 0; c < feature->count; c++){
        // If not in vocab, continue
        if(feature->index[c] == -1) continue;

        // Write the vector
        offset = feature->index[c] * layer1_size;
        for (i = 0; i < layer1_size; i++) 
            mean[i] += syn0[offset + i];

        // Increase the count
        actualCount++;
    }

    // Normalizing if non-zero count
    if(actualCount)
        for (i = 0; i < layer1_size; i++)
            mean[i] = mean[i]/actualCount;

    // Saving the embedding in the featureWord
    for(i = 0; i < layer1_size; i++)
        feature->embed[i] = mean[i];

    // Compute the magnitude of mean
    float magnitude = 0;
    for(i = 0; i < layer1_size; i++)
        magnitude += mean[i] * mean[i];
        
    //feature->magnitude = (float*) malloc(sizeof(float));
    feature->magnitude = sqrt(magnitude);

    free(mean);
}

// Compute embedding for a feature word for multi-model
void computeMultiFeatureEmbedding(struct featureWord* feature){
    // Go through the current feature and get the mean of components
    int i, c, actualCount = 0;
    long long offset;
    float *meanR, *meanS, *meanP;
    
    // Initializing with zeros
    meanR = (float*) calloc(layer1_size, sizeof(float));
    meanS = (float*) calloc(layer1_size, sizeof(float));
    meanP = (float*) calloc(layer1_size, sizeof(float));

    // Get the mean feature for the word
    for(c = 0; c < feature->count; c++){
        // If not in vocab, continue
        if(feature->index[c] == -1) continue;

        // Write the vector
        offset = feature->index[c] * layer1_size;
        for (i = 0; i < layer1_size; i++) {
            meanR[i] += syn0R[offset + i];
            meanP[i] += syn0P[offset + i];
            meanS[i] += syn0S[offset + i];
        }

        // Increase the count
        actualCount++;
    }

    // Normalizing if non-zero count
    if(actualCount)
        for (i = 0; i < layer1_size; i++){
            meanR[i] = meanR[i]/actualCount;
            meanS[i] = meanS[i]/actualCount;
            meanP[i] = meanP[i]/actualCount;
        }
    // Saving the embedding in the featureWord
    for(i = 0; i < layer1_size; i++){
        feature->embedR[i] = meanR[i];
        feature->embedP[i] = meanP[i];
        feature->embedS[i] = meanS[i];
    }

    // Compute the magnitude of meanR, meanS, meanP
    float magnitude = 0;
    for(i = 0; i < layer1_size; i++)
        magnitude += meanR[i] * meanR[i];
    feature->magnitudeR = sqrt(magnitude);

    magnitude = 0;
    for(i = 0; i < layer1_size; i++)
        magnitude += meanP[i] * meanP[i];
    feature->magnitudeP = sqrt(magnitude);

    magnitude = 0;
    for(i = 0; i < layer1_size; i++)
        magnitude += meanS[i] * meanS[i];
    feature->magnitudeS = sqrt(magnitude);

    free(meanR);
    free(meanP);
    free(meanS);
}

// Searching a feature word
int searchFeatureWord(char* word){
    unsigned int hash = getFeatureWordHash(word);

    while (1){
        if (featHashInd[hash] == -1) {
            return -1;
        }
        if (!strcmp(word, featHashWords[featHashInd[hash]].str)){
            return featHashInd[hash];
        }
        hash = (hash + 1) % featHashSize;
    }

    return -1;
}

// Adding a feature word
int addFeatureWord(char* word){
    // search for feature if already exists
    int featureInd = searchFeatureWord(word);
    
    // If yes, ignore
    if(featureInd != -1) 
        return featureInd;
    else{
        // If no, add and re-adjust featVocabSize and featVocabMaxSize
        // adding a new featureWord
        unsigned int hash = getFeatureWordHash(word);

        // Get the index where new feature word should be stored
        while(1){
            if(featHashInd[hash] != -1) 
                hash = (hash + 1) % featHashSize;
            else
                break;
        }
        // Add the word and increase vocab size
        featHashInd[hash] = featVocabSize;
        featHashWords[featVocabSize] = constructFeatureWord(word);
        featVocabSize++;

        // Adjusting the size of vocab if needed
        if(featVocabSize + 2 > featVocabMaxSize){
            featVocabMaxSize += 1000;
            featHashWords = (struct featureWord *) realloc(featHashWords, 
                                        featVocabMaxSize * sizeof(struct featureWord));
        }

        return featHashInd[hash];
    }
}

// Hash function computation
int getFeatureWordHash(char* word){
    unsigned long long a, hash = 0;
    for (a = 0; a < strlen(word); a++)
        hash = hash * 257 + word[a];
    hash = hash % featHashSize;
    return hash;
}

// Multiple character split
// Source: http://stackoverflow.com/questions/29788983/split-char-string-with-multi-character-delimiter-in-c
char *multi_tok(char *input, char *delimiter) {
    static char *string;
    if (input != NULL)
        string = input;

    if (string == NULL)
        return string;

    char *end = strstr(string, delimiter);
    if (end == NULL) {
        char *temp = string;
        string = NULL;
        return temp;
    }

    char *temp = string;

    *end = '\0';
    string = end + strlen(delimiter);
    return temp;
}

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeatures(int clusters){
    int k = clusters;                           /* number of cluster to create */
    int d = visualFeatSize;                           /* dimensionality of the vectors */
    int n = noTrain;                         /* number of vectors */
    //int nt = 1;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */
    
    // Populate the features
    float * v = fvec_new (d * n);    /* random set of vectors */
    long i, j, offset;
    for (i = 0; i < n; i++){
        offset = i * d;
        for(j = 0; j < d; j++)
            v[offset + j] = (float) train[i].feat[j];
    }
    
    /* variables are allocated externaly */
    float * centroids = fvec_new (d * k); /* output: centroids */
    float * dis = fvec_new (n);           /* point-to-cluster distance */
    int * assign = ivec_new (n);          /* quantization index of each point */
    int * nassign = ivec_new (k);         /* output: number of vectors assigned to each centroid */
    
    double t1 = getmillisecs();
    // Cluster the features
    kmeans (d, n, k, niter, v, 1, 1, redo, centroids, dis, assign, nassign);
    double t2 = getmillisecs();
    
    printf ("kmeans performed in %.3fs\n\n", (t2 - t1)  / 1000);
    //ivec_print (nassign, k);
    
    // Write the cluster ids to the prsTuple structure
    for (i = 0; i < n; i++)
        train[i].cId = assign[i] + 1;
    
    // Debugging the cId for the train tuples
    /*for (i = 0; i < n; i++)
     printf("%i\n", train[i].cId);*/
    
    // Assigning the number of clusters
    if(noClusters == 0){
        noClusters = clusters;
    }
    
    // Free memory
    free(v); free(centroids); free(dis); free(assign); free(nassign);
}

// Common sense evaluation
int performCommonSenseTask(float* testTupleScores){
    printf("Common sense task\n\n");
    // Read the validation and test sets    
    char testFile[] = "/home/satwik/VisualWord2Vec/data/test_features.txt";
    char valFile[] = "/home/satwik/VisualWord2Vec/data/val_features.txt";

    // Keep a local copy of the test scores for the best model based on threshold
    float* bestTestScore = (float*) malloc(sizeof(float) * noTest);

    if(noTest == 0 || noVal == 0)
        // Clean the strings for test and validation sets, store features
        readTestValFiles(valFile, testFile);

    // Get the features for test and validation sets
    // Re-evaluate the features for the entire vocab
    computeEmbeddings();

    // Evaluate the cosine distance
    evaluateCosDistance();

    // Going through all the test / validation examples
    // For each, going through training instances and computing the score
    valScore = (float*) malloc(noVal * sizeof(float));
    testScore = (float*) malloc(noTest * sizeof(float));

    // Threshold sweeping for validation
    // and get the best validation and correspoding testing accuracy
    float bestValAcc = 0, bestTestAcc = 0;
    float threshold;
    for(threshold = -1.0; threshold < 2.0; threshold += 0.1){
        
        computeTestValScores(val, noVal, threshold, valScore);
        computeTestValScores(test, noTest, threshold, testScore);

        // Compute the accuracy
        float* precVal = computeMAP(valScore, val, noVal);
        float* precTest = computeMAP(testScore, test, noTest);

        // Get the maximum
        if(bestValAcc < precVal[0]){
            bestValAcc = precVal[0];
            bestTestAcc = precTest[0];
            //Also store the scores for all test tuples
            if(testTupleScores != NULL){
                memcpy(bestTestScore, testScore, sizeof(float) * noTest);
            }
        }
        if(verbose)
            printf("Precision (threshold , val , test) : %f %f %f\n", 
                                        threshold, precVal[0], precTest[0]);
    }
    printf("Precision (val, test) : %f %f\n", bestValAcc, bestTestAcc);

    // Stop the procedure if validation accuracy decreases
    if(prevValAcc > bestValAcc){
        return 0;
    }
    else{
        prevValAcc = bestValAcc;
        prevTestAcc = bestTestAcc;
        // Copy the best test scores to gain access outside the function for visualizations
        if(testTupleScores != NULL)
            memcpy(testTupleScores, bestTestScore, sizeof(float) * noTest);
        return 1;
    }
    free(bestTestScore);
}

// Common sense evaluation
int performMultiCommonSenseTask(float* testTupleScores){
    printf("Common sense task with multi models....\n\n");
    // Read the validation and test sets    
    char testFile[] = "/home/satwik/VisualWord2Vec/data/test_features.txt";
    char valFile[] = "/home/satwik/VisualWord2Vec/data/val_features.txt";
    
    if(noTest == 0 || noVal == 0)
        // Clean the strings for test and validation sets, store features
        readTestValFiles(valFile, testFile);

    // Get the features for test and validation sets
    // Re-evaluate the features for the entire vocab
    computeMultiEmbeddings();

    // Evaluate the cosine distance
    evaluateMultiCosDistance();

    // Going through all the test / validation examples
    // For each, going through training instances and computing the score
    valScore = (float*) malloc(noVal * sizeof(float));
    testScore = (float*) malloc(noTest * sizeof(float));
    // Keep a local copy of the test scores for the best model based on threshold
    float* bestTestScore = (float*) malloc(sizeof(float) * noTest);

    // Threshold sweeping for validation
    // and get the best validation and correspoding testing accuracy
    float bestValAcc = 0, bestTestAcc = 0;
    float threshold;
    //float threshold, *iterPrecVal, *iterPrecTest;
    int i;
    int *randPermVal = (int*) malloc(sizeof(int) * noVal);
    int *randPermTest = (int*) malloc(sizeof(int) * noVal);

    //float precTest = (float*) malloc(sizeof(float) * 
    //float precTest[2], precVal[2], iterPrecTest[2], iterPrecVal[2];
    float* precVal = (float*) malloc(sizeof(float) * 2);
    float* precTest = (float*) malloc(sizeof(float) * 2);
    float* iterPrecTest = (float*) malloc(sizeof(float) * 2);
    float* iterPrecVal = (float*) malloc(sizeof(float) * 2);
    for(threshold = -1.0; threshold < 2.0; threshold += 0.1){
        
        computeMultiTestValScores(val, noVal, threshold, valScore);
        computeMultiTestValScores(test, noTest, threshold, testScore);

        // Compute the accuracy
        //precVal = computeMAP(valScore, val, noVal);
        //precTest = computeMAP(testScore, test, noTest);

        precVal[0] = 0; precVal[1] = 0;
        precTest[0] = 0; precTest[1] = 0;
        // Compute the accuracy for multiple permutations
        int noIters = 100;
        for (i = 0; i < noIters; i++){
            printf("%d ", i);
            // generate a permutation
            rpermute(noVal, randPermVal); 
            iterPrecVal = computePermuteMAP(valScore, val, randPermVal, noVal);

            rpermute(noTest, randPermTest); 
            iterPrecTest = computePermuteMAP(testScore, test, randPermTest, noTest);

            // Updating the precVal, precTest
            precVal[0] += iterPrecVal[0];
            precVal[1] += iterPrecVal[1];
            precTest[0] += iterPrecTest[0];
            precTest[1] += iterPrecTest[1];
        }
        printf("\n");
        // Normalizing for the mean
        precVal[0] = precVal[0] / noIters;
        precVal[1] = precVal[1] / noIters;
        precTest[0] = precTest[0] / noIters;
        precTest[1] = precTest[1] / noIters;

        // Get the maximum
        if(bestValAcc < precVal[0]){
            bestValAcc = precVal[0];
            bestTestAcc = precTest[0];
            //Also store the scores for all test tuples
            if(testTupleScores != NULL)
                memcpy(bestTestScore, testScore, sizeof(float) * noTest);
        }
        if(verbose)
            printf("Precision (threshold , val , test) : %f %f %f\n", 
                                        threshold, precVal[0], precTest[0]);
    }
    printf("Precision (val, test) : %f %f\n", bestValAcc, bestTestAcc);

    // Stop the procedure if validation accuracy decreases
    if(prevValAcc > bestValAcc){
        return 0;
    }
    else{
        prevValAcc = bestValAcc;
        prevTestAcc = bestTestAcc;
        // Copy the best test scores to gain access outside the function for visualizations
        if(testTupleScores != NULL)
            memcpy(testTupleScores, bestTestScore, sizeof(float) * noTest);

        return 1;
    }
    free(bestTestScore);
    free(randPermVal);
    free(randPermTest);
}

// Reading the test and validation files
void readTestValFiles(char* valName, char* testName){
    // Read the file
    long noTuples = 0, i;
    
    // Counting the number of lines
    char pWord[MAX_STRING], 
         rWord[MAX_STRING], 
         sWord[MAX_STRING];
    int gTruth = -1;

    FILE* filePt = fopen(valName, "rb");
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]> %d\n", pWord, rWord, sWord, &gTruth) != EOF)
        noTuples++;

    // Rewind the stream and read again
    rewind(filePt);
    
    // Initialize and save the feature words
    val = (struct prsTuple*) malloc(sizeof(struct prsTuple) * noTuples);

    for(i = 0; i < noTuples; i++){
        if(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]> %d\n", pWord, rWord, sWord, &gTruth) != EOF){
            val[i].p = addFeatureWord(pWord);
            val[i].r = addFeatureWord(rWord);
            val[i].s = addFeatureWord(sWord);
        
            val[i].cId = gTruth;
            //printf("%d = <%d:%d:%d> %d\n", i, val[i].p, val[i].r, val[i].s, val[i].cId);
        }
    }

    noVal = noTuples;
    printf("Found %ld tuples in %s...\n\n", noTuples, valName);
    // Close the file
    fclose(filePt);
    /*******************************************************************************/
    // Test file
    // filePt 
    filePt = fopen(testName, "rb");
    noTuples = 0;
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]> %d\n", pWord, rWord, sWord, &gTruth) != EOF)
        noTuples++;

    // Rewind the stream and read again
    rewind(filePt);
    
    // Initialize and save the feature words
    test = (struct prsTuple*) malloc(sizeof(struct prsTuple) * noTuples);
    for(i = 0; i < noTuples; i++){
        if(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]> %d\n", pWord, rWord, sWord, &gTruth) != EOF){
            test[i].p = addFeatureWord(pWord);
            test[i].r = addFeatureWord(rWord);
            test[i].s = addFeatureWord(sWord);
        
            test[i].cId = gTruth;
            //printf("%d <%d:%d:%d> %d\n", i, test[i].p, test[i].r, test[i].s, test[i].cId);
        }
    }

    noTest = noTuples;
    printf("Found %ld tuples in %s...\n\n", noTuples, testName);
    // Close the file
    fclose(filePt);
}

// Cosine distance evaluation
void evaluateCosDistance(){
    if (verbose) printf("Evaluating pairwise dotproducts..\n\n");
    // Allocate memory for cosDist variable
    cosDist = (float*) malloc(featVocabSize * featVocabSize * sizeof(float));
    
    // For each pair, we evaluate the dot product along with normalization
    long a, b, i, offset;
    float magProd = 0, dotProduct;
    for(a = 0; a < featVocabSize; a++){
        offset = featVocabSize * a;
        for(b = a; b < featVocabSize; b++){
            if(featHashWords[a].embed == NULL || featHashWords[b].embed == NULL)
                printf("NULL pointers : %ld %ld\n", a, b);

            dotProduct = 0;
            for(i = 0; i < layer1_size; i++){
                dotProduct += 
                    featHashWords[a].embed[i] * featHashWords[b].embed[i];
            }
            
            // Save the dotproduct
            magProd = (featHashWords[a].magnitude) * (featHashWords[b].magnitude);
            if(magProd){
                cosDist[offset + b] = dotProduct / magProd;
                cosDist[a + b * featVocabSize] = dotProduct / magProd;
            }
             else{
                cosDist[offset + b] = 0.0;
                cosDist[a + b * featVocabSize] = 0.0;
            }
        }
    }
}

// Cosine distance evaluation for multi model
void evaluateMultiCosDistance(){
    if (verbose) printf("Evaluating pairwise dot products(multi)....\n\n");
    // Allocate memory for cosDist variable
    cosDistR = (float*) malloc(featVocabSize * featVocabSize * sizeof(float));
    cosDistS = (float*) malloc(featVocabSize * featVocabSize * sizeof(float));
    cosDistP = (float*) malloc(featVocabSize * featVocabSize * sizeof(float));
    
    // For each pair, we evaluate the dot product along with normalization
    long a, b, i, offset;
    float magProd = 0, dotProduct;
    for(a = 0; a < featVocabSize; a++){
        offset = featVocabSize * a;
        for(b = a; b < featVocabSize; b++){
            // cosDist for P
            if(featHashWords[a].embedP == NULL || featHashWords[b].embedP == NULL)
                printf("NULL pointers : %ld %ld\n", a, b);

            dotProduct = 0;
            for(i = 0; i < layer1_size; i++){
                dotProduct += 
                    featHashWords[a].embedP[i] * featHashWords[b].embedP[i];
            }
            
            // Save the dotproduct (symmetrically)
            magProd = (featHashWords[a].magnitudeP) * (featHashWords[b].magnitudeP);
            if(magProd){
                cosDistP[offset + b] = dotProduct / magProd;
                cosDistP[a + b * featVocabSize] = dotProduct / magProd;
            }
             else{
                cosDistP[offset + b] = 0.0;
                cosDistP[a + b * featVocabSize] = 0.0;
            }
            // ===================================================================//
            // cosDist for R
            if(featHashWords[a].embedR == NULL || featHashWords[b].embedR == NULL)
                printf("NULL pointers : %ld %ld\n", a, b);

            dotProduct = 0;
            for(i = 0; i < layer1_size; i++){
                dotProduct += 
                    featHashWords[a].embedR[i] * featHashWords[b].embedR[i];
            }
            
            // Save the dotproduct
            magProd = (featHashWords[a].magnitudeR) * (featHashWords[b].magnitudeR);
            if(magProd){
                cosDistR[offset + b] = dotProduct / magProd;
                cosDistR[a + b * featVocabSize] = dotProduct / magProd;
            }
            else{
                cosDistR[offset + b] = 0.0;
                cosDistR[a + b * featVocabSize] = 0.0;
            }
            // ===================================================================//
            // cosDist for S
            if(featHashWords[a].embedS == NULL || featHashWords[b].embedS == NULL)
                printf("NULL pointers : %ld %ld\n", a, b);

            dotProduct = 0;
            for(i = 0; i < layer1_size; i++){
                dotProduct += 
                    featHashWords[a].embedS[i] * featHashWords[b].embedS[i];
            }
            
            // Save the dotproduct
            magProd = (featHashWords[a].magnitudeS) * (featHashWords[b].magnitudeS);
            if(magProd){
                cosDistS[offset + b] = dotProduct / magProd;
                cosDistS[a + b * featVocabSize] = dotProduct / magProd;
            }
            else{
                cosDistS[offset + b] = 0.0;
                cosDistS[a + b * featVocabSize] = 0.0;
            }
        }
    }
}

// Computing the test and validation scores
void computeTestValScores(struct prsTuple* holder, long noInst, float threshold, float* scoreList){
    if(verbose) printf("Computing the scores...\n\n");

    // Iteration variables
    long a, b;
    float meanScore, pScore, rScore, sScore, curScore; 
    for(a = 0; a < noInst; a++){
        meanScore = 0.0;
        // For each training instance, find score, ReLU and max
        for(b = 0; b < noTrain; b++){
            // Get P score
            pScore = cosDist[featVocabSize * train[b].p + holder[a].p];
            
            // Get R score
            rScore = cosDist[featVocabSize * train[b].r + holder[a].r];
            
            // Get S score
            sScore = cosDist[featVocabSize * train[b].s + holder[a].s];
           
            // ReLU
            curScore = pScore + rScore + sScore - threshold;
            if(curScore < 0) curScore = 0;
            
            // Add it to the meanScore
            meanScore += curScore;
        }

        // Save the mean score for the current instance
        scoreList[a] = meanScore / noTrain;
        //printf("%ld : %f\n", a, scoreList[a]);
    }
}

// Computing the test and validation scores
void computeMultiTestValScores(struct prsTuple* holder, long noInst, float threshold, float* scoreList){
    if(verbose) printf("Computing the scores...\n\n");

    // Iteration variables
    long a, b;
    float meanScore, pScore, rScore, sScore, curScore; 
    for(a = 0; a < noInst; a++){
        meanScore = 0.0;
        // For each training instance, find score, ReLU and max
        for(b = 0; b < noTrain; b++){
            // Get P score
            pScore = cosDistP[featVocabSize * train[b].p + holder[a].p];
            
            // Get R score
            rScore = cosDistR[featVocabSize * train[b].r + holder[a].r];
            
            // Get S score
            sScore = cosDistS[featVocabSize * train[b].s + holder[a].s];
           
            // ReLU
            curScore = pScore + rScore + sScore - threshold;
            if(curScore < 0) curScore = 0;
            
            // Add it to the meanScore
            meanScore += curScore;
        }

        // Save the mean score for the current instance
        scoreList[a] = meanScore / noTrain;
        //printf("%ld : %f\n", a, scoreList[a]);
    }
}

// Compute mAP and basic precision
float* computeMAP(float* score, struct prsTuple* holder, long noInst){
    if (verbose) printf("Computing MAP...\n\n");
    // Crude implementation
    long a, b;
    int* rankedLabels = (int*) malloc(sizeof(int) * noInst);
    float* rankedScores = (float*) malloc(sizeof(float) * noInst);

    // Make a copy of scores
    for(a = 0; a < noInst; a++) rankedScores[a] = score[a];

    long maxInd;
    // Get the rank of positive instances wrt to the scores
    for(a = 0; a < noInst; a++){
        maxInd = 0;
        for(b = 0; b < noInst; b++)
            // Check if max is also current max (flag out using -5)
            if(rankedScores[b] != -5 && 
                        rankedScores[maxInd] < rankedScores[b]) 
                maxInd = b;

        // Swap the max and element at that instance
        rankedLabels[a] = holder[maxInd].cId;
        // NULLing the max ind 
        rankedScores[maxInd] = -5;
    }

    float mAP = 0, base = 0;
    long noPositives = 0;
    // Compute the similarity wrt ideal ordering 1,2,3.....
    for(a = 0; a < noInst; a++)
        if(rankedLabels[a] == 1){
            // Increasing the positives
            noPositives++;
            mAP += noPositives / (float) (a + 1);
            //printf("%ld %ld %f\n", noPositives, a + 1, noPositives/(float) (a + 1));
        }
   
    // Compute mAP
    mAP = mAP / noPositives;
    // Compute base precision
    base = noPositives / (float)noInst;
    // Packing both in an array
    float* precision = (float*) malloc(2 * sizeof(float));
    precision[0] = mAP;
    precision[1] = base;

    // Free memory
    free(rankedScores);
    free(rankedLabels);
    return precision;
}

// Compute mAP and basic precision for a permutation of the data specified
float* computePermuteMAP(float* score, struct prsTuple* holder, int* permute, long noInst){
    if (verbose) printf("Computing MAP...\n\n");
    // Crude implementation
    long a, b;
    int* rankedLabels = (int*) malloc(sizeof(int) * noInst);
    float* rankedScores = (float*) malloc(sizeof(float) * noInst);

    // Make a copy of scores
    for(a = 0; a < noInst; a++) rankedScores[a] = score[a];

    long maxInd;
    // Get the rank of positive instances wrt to the scores
    for(a = 0; a < noInst; a++){
        maxInd = permute[0];
        for(b = 0; b < noInst; b++)
            // Check if max is also current max
            if(rankedScores[permute[b]] != -5 && 
                        rankedScores[maxInd] < rankedScores[permute[b]]) 
                maxInd = permute[b];

        // Swap the max and element at that instance
        rankedLabels[a] = holder[maxInd].cId;
        // NULLing the max ind 
        rankedScores[maxInd] = -5;
    }

    float mAP = 0, base = 0;
    long noPositives = 0;
    // Compute the similarity wrt ideal ordering 1,2,3.....
    for(a = 0; a < noInst; a++)
        if(rankedLabels[a] == 1){
            // Increasing the positives
            noPositives++;
            mAP += noPositives / (float) (a + 1);
            //printf("%ld %ld %f\n", noPositives, a + 1, noPositives/(float) (a + 1));
        }
   
    // Compute mAP
    mAP = mAP / noPositives;
    // Compute base precision
    base = noPositives / (float)noInst;
    // Packing both in an array
    float* precision = (float*) malloc(2 * sizeof(float));
    precision[0] = mAP;
    precision[1] = base;

    // Free memory
    free(rankedScores);
    free(rankedLabels);
    return precision;
}

// Saving the word2vec vectors for further use
void saveWord2Vec(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    long i, j, offset;
    for (i = 0; i < vocab_size; i++){
        offset = i * layer1_size;
        fprintf(filePt, "%s ", vocab[i].word);
        for (j = 0; j < layer1_size - 1; j++)
            fprintf(filePt, "%f ", syn0[offset + j]);

        fprintf(filePt, "%f\n", syn0[offset + layer1_size - 1]);
    }

    fclose(filePt);
}

// Load the word2vec vectors
// We assume all the other parameters(vocabulary is kept constant)
// Use with caution
void loadWord2Vec(char* fileName){
    FILE* filePt = fopen(fileName, "rb");
    
    long i, j, offset;
    long noVocab, dims;
    float value;
    char word[MAX_STRING];
    fscanf(filePt, "%ld %ld\n", &noVocab, &dims);
    if(vocab_size != noVocab && layer1_size != dims){
        printf("Word2Vec reading incompatible! \n");
        exit(1);
    }

    // Reading the dimensions
    for (i = 0; i < vocab_size; i++){
        fscanf(filePt, "%s", word);
        // Allocate memory and store the word
        offset = layer1_size * i;
        for(j = 0; j < layer1_size; j++){
            fscanf(filePt, "%f", &value);

            // Storing the value
            syn0[offset + j] = value;
        }
        //printf("%s\n", word);
    }

    fclose(filePt);
}

/////////////////////////////////////////////////////////////////////////////////////
// Analysis
// FInd the best test tuple with maximum improvements
void findBestTestTuple(float* baseScore, float* bestScore){
    // Initialize list of improved test tuple and count
    int* improvedInd = (int*) malloc(sizeof(int) * noTest);
    int count = 0;

    // Check if the tuple if positive and there is an increase
    long i;
    for(i = 0; i < noTest; i++){
        if(test[i].cId && (bestScore[i] > baseScore[i])){
            // Store the index and increase the count
            improvedInd[count] = i;
            count++;
        }
    }

    printf("%d tuples improved! \n", count);

    // Do something here
    // Dump the tuples and embeddings along with base and best score
    char tupleFile[] = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/improved_test_tuples.txt";
    char embedFile[] = "/home/satwik/VisualWord2Vec/code/word2vecVisual/modelsNdata/improved_test_embed.txt";

    saveMultiTupleEmbeddings(tupleFile, embedFile, test, baseScore, bestScore, improvedInd, count);

    free(improvedInd);
}
