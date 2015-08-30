# include "visualFeatures.h"

// Storing the feature hash (globals)
struct featureWord* featHashWords;
int* featHashInd;
const int featHashSize = 100000;
int featVocabSize = 0;
int featVocabMaxSize = 2000;
/***************************************************************************/
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
        //printf("%s : %s : %s\n", pWord, sWord, rWord);
        
        // Getting the indices for p, s, r
        // Get the indices, for the current tuple
        prs[noTuples].p = addFeatureWord(pWord);
        prs[noTuples].r = addFeatureWord(rWord);
        prs[noTuples].s = addFeatureWord(sWord);
        //printf("%s : %s : %s\n", prs[noTuples].p.str, prs[noTuples].s.str, prs[noTuples].r.str);
        
        noTuples++;
    }

    // Debugging
    /*int i;
    for(i = 0; i < noTuples; i++){
        //printf("%s : %s : %s  ( ", prs[i].p.str, prs[i].s.str, prs[i].r.str);
        //printf("%d )\n", i);
        //printf("%d %d %d\n", prs[i].p, prs[i].s, prs[i].r);
        printf("%s : %s : %s\n", featHashWords[prs[i].p].str, 
                            featHashWords[prs[i].s].str,
                            featHashWords[prs[i].r].str);
    }*/

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
        //if(prs[i].cId == -1) prs[i].cId = clusterId;
        prs[i].cId = clusterId;
        i++;
    }

    // Sanity check
    if(i != NUM_TRAINING){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }

    // Debugging
    /*for(i = 0; i < NUM_TRAINING; i++){
        printf("%s : %s : %s : (%d)\n", prs[i].p.str, prs[i].s.str, prs[i].r.str, prs[i].cId);
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

    int feature, i, noLines = 0;
    // Reading till EOF
    while(fscanf(filePt, "%d", &feature) != EOF){
        prs[noLines].feat = (int*) malloc(sizeof(int) * VISUAL_FEATURE_SIZE);
        prs[noLines].feat[0] = feature;

        for(i = 1; i < VISUAL_FEATURE_SIZE; i++){
            //printf("%d ", feature);
            fscanf(filePt, "%d", &feature);
            prs[noLines].feat[i] = feature;
        }
        //printf("%d\n", feature);

        //printf("Line : %d\n", noLines);
        noLines++;
    }

    // Closing the file
    fclose(filePt);
}

// Finding the indices of words for P,R,S
struct featureWord constructFeatureWord(char* word){
    int index = SearchVocab(word); 

    struct featureWord feature;
    feature.str = (char*) malloc(MAX_STRING_LENGTH);
    strcpy(feature.str, word);
    
    // Do something if not in vocab
    if(index == -1) {
        //printf("Not in vocab -> %s : %s\n", word, "") ;
        int count=0, i;

        // Split based on 's
        char* token = (char*) malloc(MAX_STRING_LENGTH);
        strcpy(token, word);

        char* first = multi_tok(token, "'s");
        char* second = multi_tok(NULL, "'s");

        // Join both the parts without the 's
        if(second != NULL) token = strcat(first, second);
        else token = first;

        char* temp = (char*) malloc(MAX_STRING_LENGTH);
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

// Initializing the refining
void initRefining(){
    long long a, b;
    unsigned long long next_random = 1;

    // Setup the network 
    a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
    if (syn1 == NULL) {
        printf("Memory allocation failed\n"); 
        exit(1);
    }

    // Initialize the last layer of weights
    for (a = 0; a < NUM_CLUSTERS; a++) for (b = 0; b < layer1_size; b++){
        next_random = next_random * (unsigned long long)25214903917 + 11;
        syn1[a * layer1_size + b] = (((next_random & 0xFFFF) / (real)65536) - 0.5) / layer1_size;
    }

    // Setting up the hash
    featHashWords = (struct featureWord *) malloc(sizeof(struct featureWord) * featVocabMaxSize);
    featHashInd = (int*) malloc(sizeof(int) * featHashSize);
    for(a = 0; a < featHashSize; a++)
        featHashInd[a] = -1;
}

// Refine the network through clusters
void refineNetwork(){
    // Reading the features for debugging
    /*int x, z;
    for(x = 0; x < NUM_TRAINING; x++){
        for( z = 0; z < VISUAL_FEATURE_SIZE; z++){
            printf("%d ", prs[x].feat[z]);
        }
        printf("\n");
    }*/

    long long c, i;
    float* y = (float*) malloc(sizeof(float) * NUM_CLUSTERS);
    struct featureWord p, s, r;

    // Read each of the training instance
    for(i = 0; i < NUM_TRAINING; i++){
        printf("Training %lld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(prs[i].cId < 1 || prs[i].cId > NUM_CLUSTERS) {
            printf("\nCluster id (%d) for %lld instance invalid!\n", prs[i].cId, i);
            exit(1);
        }

        //printf("Counts : %d %d %d\n", prs[i].p.count, prs[i].s.count, prs[i].r.count);

        // Updating the weights for P
        p = featHashWords[prs[i].p];
        for(c = 0; c < p.count; c++){
            // If not in vocab, continue
            if(p.index[c] == -1) continue;
            //printf("p: %d %d\n", p.index[c], p.count);

            // Predict the cluster
            computeMultinomial(y, p.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, p.index[c], prs[i].cId);
        }
        
        // Updating the weights for S
        s = featHashWords[prs[i].s];
        for(c = 0; c < s.count; c++){
            // If not in vocab, continue
            if(s.index[c] == -1) continue;
            //printf("s: %d %d\n", s.index[c], s.count);

            // Predict the cluster
            computeMultinomial(y, s.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, s.index[c], prs[i].cId);
        }

        // Updating the weights for R
        r = featHashWords[prs[i].r];
        for(c = 0; c < r.count; c++){
            // If not in vocab, continue
            if(r.index[c] == -1) continue;
            //printf("r: %d %d\n", r.index[c], r.count);

            // Predict the cluster
            computeMultinomial(y, r.index[c]);
            // Propage the error to the PRS features
            updateWeights(y, r.index[c], prs[i].cId);
        }
    }
}

// Evaluate y_i for each output cluster
void computeMultinomial(float* y, int wordId){
    // y stores the multinomial distribution
    float dotProduct = 0, sum = 0;
    long long a, b, offset1, offset2;

    // Offset to access the outer layer weights
    offset1 = wordId * layer1_size;
    for (b = 0; b < NUM_CLUSTERS; b++){
        dotProduct = 0;
        // Offset to access the values of hidden layer weights
        offset2 = b * layer1_size;

        for (a = 0; a < layer1_size; a++){
            dotProduct += syn0[offset1 + a] * syn1[offset2 + a];
        }

        // Exponential (clip if less or greater than the limit)
        if (dotProduct <= - MAX_EXP) dotProduct = -MAX_EXP;
        else if (dotProduct >= MAX_EXP) dotProduct = MAX_EXP;
        else dotProduct = expTable[(int) ((dotProduct + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];

        y[b] = dotProduct;
    }

    // Normalizing to create a probability measure
    for(b = 0; b < NUM_CLUSTERS; b++) sum += y[b];

    if(sum > 0)
        for(b = 0; b < NUM_CLUSTERS; b++) y[b] = y[b]/sum;
}

// Updating the weights given the multinomial prediction, word id and true cluster id
void updateWeights(float* y, int wordId, int trueId){
    // compute gradient for outer layer weights, gradient g
    float* e = (float*) malloc(NUM_CLUSTERS * sizeof(float));
    long long a, b, c, offset1, offset2;
    float learningRate = 0.01;

    // Computing error
    for(b = 0; b < NUM_CLUSTERS; b++){
        if(b == trueId - 1) e[b] = y[b] - 1;
        else e[b] = y[b];
    }
    // compute gradient for inner layer weights
    // update inner layer weights
    // Offset for accessing inner weights
    offset1 = layer1_size * wordId;
    for(b = 0; b < NUM_CLUSTERS; b++){
        // Offset for accesing outer weights
        offset2 = layer1_size * b;
        
        for(c = 0; c < layer1_size; c++)
            syn0[offset1 + c] -= learningRate * e[b] * syn1[offset2 + c];
    }

    // compute gradient for outer layer weights
    // update outer layer weights
    offset1 = layer1_size * wordId;
    for(a = 0; a < NUM_CLUSTERS; a++){
        offset2 = layer1_size * a;
        for(b = 0; b < layer1_size; b++){
            syn1[offset2 + b] -= learningRate * e[a] * syn0[offset1 + b];
        }
    }

    free(e);
}

// Saving the feature embeddings needed for comparing, at the given file name
void saveEmbeddings(char* saveName){
    FILE* filePt = fopen(saveName, "wb");
    int i;

    // Go through the vocab and save the embeddings
    for(i = 0; i < featVocabSize; i++)
        saveFeatureEmbedding(featHashWords[i], filePt);

    fclose(filePt);
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

// Save a particular embedding
void saveFeatureEmbedding(struct featureWord feature, FILE* filePt){
    // Go through the current feature and get the mean of components
    int i, c, actualCount = 0;
    long long offset;
    float* mean;
    mean = (float*) calloc(layer1_size, sizeof(float));

    // Get the mean feature for the word
    for(c = 0; c < feature.count; c++){
        // If not in vocab, continue
        if(feature.index[c] == -1) continue;

        // Write the vector
        offset = feature.index[c] * layer1_size;
        for (i = 0; i < layer1_size; i++) 
            mean[i] += syn0[offset + i];

        // Increase the count
        actualCount++;
    }

    // Normalizing if non-zero count
    if(actualCount)
        for (i = 0; i < layer1_size; i++)
            mean[i] = mean[i]/actualCount;

    // Saving to the file
    //fprintf(filePt, "%s\n", feature.str);
    for(i = 0; i < layer1_size-1; i++)
        fprintf(filePt, "%f ", mean[i]);
    fprintf(filePt, "%f\n", mean[layer1_size-1]);
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
void clusterVisualFeatures(int noClusters){
    int k = noClusters;                           /* number of cluster to create */
    int d = VISUAL_FEATURE_SIZE;                           /* dimensionality of the vectors */
    int n = NUM_TRAINING;                         /* number of vectors */
    int nt = 1;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */

    // Populate the features
    float * v = fvec_new (d * n);    /* random set of vectors */
    long i, j, offset;
    for (i = 0; i < n; i++){
        offset = i * d;
        for(j = 0; j < d; j++)
            v[offset + j] = (float) prs[i].feat[j];
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
        prs[i].cId = assign[i];

    // Debugging the cId for the prs tuples
    /*for (i = 0; i < n; i++)
        printf("%i\n", prs[i].cId);

    // Free memory
    free(v); free(centroids); free(dis); free(assign); free(nassign);
}
