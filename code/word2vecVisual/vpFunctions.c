// Definitions for functions related to the visual paraphrasing task
# include "vpFunctions.h"

// Globals for the current scope
long noTrainVP = 0;
int vpFeatSize = 0;
int otherFeatSize = 0; // The size of the other features
struct Sentence* trainSents; // Training sentences (mixed of set1, set2 + training + gt==1)
struct Sentence* sentences1; // Set of sentences 1
struct Sentence* sentences2; // Set of sentences 2 
struct SentencePair* sentPairs; // Dataset of pairs of sentences
long noSentPairs; // Number of sentences pairs

// Training the sentences
// Could be one of DESCRIPTIONS, SENTENCES, WORDS, WINDOWS;
enum TrainModeVP mode = WINDOWS;

/***************************************************/
// Read the sentences
struct Sentence** readSentences(char* featurePath, long* noSents){
    // Open the file
    FILE* filePtr = fopen(featurePath, "rb");

    if(filePtr == NULL){
        printf("File not found !\n");
        exit(1);
    }

    // Buffer to store the current sentence
    char* currentSent = (char*) malloc(sizeof(char) * MAX_SENTENCE);

    long sentCount = 0, i;
    // Read lines one by one, just counting number of sentences
    while(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL)
        sentCount++;

    // Rewind the stream and read again
    rewind(filePtr);
    
    // Allocate the collection
    struct Sentence* collection = 
                (struct Sentence*) malloc(sizeof(struct Sentence) * sentCount);
    // Read and store contents
    for( i = 0; i < sentCount; i++){
        if(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL){
            // Remove the trailing \n
            currentSent[strlen(currentSent) - 1] = '\0';
            
            // Allocate memory and copy sentence
            collection[i].sent = (char*) malloc(sizeof(char) * MAX_SENTENCE);
            strcpy(collection[i].sent, currentSent);
        }

        // Assign other members as needed
        collection[i].embed = NULL;

        // Assign some gt
        collection[i].gt = -1;
    }
    // Store the number of sentences
    *noSents = sentCount;

    fclose(filePtr);
    printf("\nFile read with %ld sentences!....\n", sentCount);
    
    ////////////////////////////////////////////////////////////////////
    // Writing it back to the file for debugging
    /*FILE* savePtr = fopen("vp_train_emit.txt", "wb");
    for (i = 0; i < sentCount; i++)
        fprintf(savePtr, "%s\n", collection[i].sent);
    fclose(savePtr);*/

    return &collection;
}

// Tokenize sentences
void tokenizeSentences(struct Sentence* collection, long noSents){
    long i;
    for (i = 0; i < noSents; i++){
        //printf("**************************\n");
        // Copy the word into a local variable line
        char* line = (char*) malloc(MAX_SENTENCE);
        strcpy(line, collection[i].sent);

        int count = 0, n, actCount = 0, sentCount = 0;

        // Split based on 's
        char* first = multi_tok(line, "'s");
        char* second = multi_tok(NULL, "'s");

        // Join both the parts without the 's (from baseline: add it at the end)
        if(second != NULL) line = strcat(first, strcat(second, " \'s"));
        else line = first;

        char* temp = (char*) malloc(MAX_SENTENCE);
        strcpy(temp, line);
        
        // Remove ' ', ',', '.', '?', '!', '\', '/'
        char* delim = " ,/!?\\"; // Ignore the full stop, used to demarcate end of sentence
        line = strtok(line, delim);
        // Going over the line to determine the number of parts
        while(line != NULL){
            count++;

            // Check if an ending word
            if(line[strlen(line)-1] == '.')
                sentCount++;
            
            // Get the next word
            line = strtok(NULL, delim);
        }

        // Now store the word components, looping over them
        if(sentCount == 0) sentCount = 1; // Punctuations not present, treat as one sentence

        collection[i].index = (int*) malloc(count * sizeof(int));
        collection[i].endIndex = (int*) malloc(sentCount * sizeof(int));

        line = strtok(temp, delim);
        count = 0, sentCount = 0;
        int lineEnd;
        int wordIndex;
        while(line != NULL){
            // Convert the token into lower case
            for(n = 0; line[n]; n++){
                line[n] = tolower(line[n]);

                // Check if it has a trailing full stop, if yes, removeit and report
                if (line[n] == '.'){
                    lineEnd = 1;
                    line[n] = '\0';
                }
            }
            wordIndex = SearchVocab(line);
            // Exists in vocab, save
            if (wordIndex != -1){
                collection[i].index[count] = wordIndex;
            
                actCount++;
                count++;
            }
            
            // Adjust end of line count
            if(lineEnd){
                collection[i].endIndex[sentCount] = count;
                sentCount++;
                lineEnd = 0;
            }
            
            // Next word
            line = strtok(NULL, delim);
        }

        // Punctuations absent, treat everything as one setnence
        if(sentCount == 0){
            sentCount = 1;
            collection[i].endIndex[0] = count;
        }

        // Now store the word components, looping over them
        collection[i].count = count;
        collection[i].actCount = actCount;
        collection[i].sentCount = sentCount;

        //printf("Sent count: %s\n%d\n", collection[i].sent, collection[i].sentCount);
    }
}

// Save the sentence embeddings
void writeSentenceEmbeddings(char* saveName, struct Sentence* collection, long noSents){
    // Open the file
    FILE* filePtr = fopen(saveName, "wb");

    // Write the number of dimensions
    fprintf(filePtr, "%lld\n", layer1_size);

    // Loop and write features for each sentence
    long i, d;
    for (i = 0; i < noSents; i++){
        for(d = 0; d < layer1_size - 1; d++)
            fprintf(filePtr, "%f ", collection[i].embed[d]);
        
        fprintf(filePtr, "%f\n", collection[i].embed[layer1_size-1]);
    }

    // Close the file
    fclose(filePtr);
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

/***************************************************/
// Read the training sentences for VP task
void readVPTrainSentences(char* featurePath){
    long noSents = 0;
    // Use readSentences
    trainSents = *readSentences(featurePath, &noSents);

    if(noTrainVP != 0){
        if(noTrainVP != noSents){
            printf("Mismatch with number of training examples: %s\n", featurePath);
            exit(1);
        }
    }
    else noTrainVP = noSents;
}

// Read the visual features for VP task
void readVPAbstractVisualFeatures(char* visualPath){
    FILE* filePt = fopen(visualPath, "rb");

    if(filePt == NULL){
        printf("File not found for visual features!\n");
        exit(1);
    }

    // Read first line the dimension of visual features
    int featDim = 0;
    fscanf(filePt, "%d\n", &featDim);
    printf("\nVisual feature size : %d\n", featDim);
    
    long noLines = 0, i;
    float feature;
    while(fscanf(filePt, "%f", &feature) != EOF){
        // Allocate memory
        trainSents[noLines].vFeat = (float*) malloc(sizeof(float) * featDim);
       
        // First entry
        trainSents[noLines].vFeat[0] = feature;

        for (i = 1; i < featDim; i++){
            fscanf(filePt, "%f", &feature);
            trainSents[noLines].vFeat[i] = feature;
        }
    
        noLines++;
    }

    // Tallying with existing numbers
    if(noTrainVP != 0 && noTrainVP != noLines){
        printf("\nNumber of training sentences and visual features dont match!\n");
        printf("NoTrainVP: %ld\nNo lines: %ld\n", noTrainVP, noLines);
        exit(1);
    }
    vpFeatSize = featDim;

    // Closing the file
    fclose(filePt);
}

// Function to tokenize the training sentences and link to word2vec vocab
void tokenizeTrainSentences(){
    // Use tokenizeSentences
    tokenizeSentences(trainSents, noTrainVP);

    // Debug
    /*long i, j;
    for (i = 0; i < noTrainVP; i++){
        printf("%s\n", train[i].sent);
        for (j = 0; j < train[i].count; j++){
            printf("%ld ", trainSents[i].index[j]);
        }
        printf("\n");
    }*/
}

// Saving word2vec embeddings for all VP sentences
void writeVPSentenceEmbeddings(){
    // Path to the sentences_1
    char readSent1[] = "/home/satwik/VisualWord2Vec/data/vp_sentences1_lemma.txt";
    char writeSent1[] = "/home/satwik/VisualWord2Vec/data/vp_30_features_1.txt";
    //char writeSent1[] = "/home/satwik/VisualWord2Vec/data/vp_orig_features_1.txt";
    // Path to the sentences_2
    char readSent2[] = "/home/satwik/VisualWord2Vec/data/vp_sentences2_lemma.txt";
    char writeSent2[] = "/home/satwik/VisualWord2Vec/data/vp_30_features_2.txt";
    //char writeSent2[] = "/home/satwik/VisualWord2Vec/data/vp_orig_features_2.txt";
    
    struct Sentence* sentences1, *sentences2;
    long noSents1, noSents2;

    // read sentences
    sentences1 = *readSentences(readSent1, &noSents1);
    sentences2 = *readSentences(readSent2, &noSents2);

    // Tokenize sentences
    tokenizeSentences(sentences1, noSents1);
    tokenizeSentences(sentences2, noSents2);

    // compute embeddings
    computeSentenceEmbeddings(sentences1, noSents1);
    computeSentenceEmbeddings(sentences2, noSents2);

    // write back to file
    writeSentenceEmbeddings(writeSent1, sentences1, noSents1);
    writeSentenceEmbeddings(writeSent2, sentences2, noSents2);
}

// Read all sentences along with features
void readVPSentences(){
    // First read the sentences
    char* readSent1 = (char*) malloc(100 * sizeof(char));
    char* readSent2 = (char*) malloc(100 * sizeof(char));

    // Path to the sentences_1
    if(debugModeVP){
        readSent1 = "/home/satwik/VisualWord2Vec/data/vp_sentences1_lemma_debug.txt";
        readSent2 = "/home/satwik/VisualWord2Vec/data/vp_sentences2_lemma_debug.txt";
    }
    // Path to the sentences_2
    else{
        readSent1 = "/home/satwik/VisualWord2Vec/data/vp_sentences1_lemma.txt";
        readSent2 = "/home/satwik/VisualWord2Vec/data/vp_sentences2_lemma.txt";
    }
    
    // read sentences
    long noSents1, noSents2;
    sentences1 = *readSentences(readSent1, &noSents1);
    sentences2 = *readSentences(readSent2, &noSents2);
    printf("\nSentences for VP read!\n");
    if(noSents1 != noSents2){
        printf("Number of sentences dont match!\n");
        exit(1);
    }
    noSentPairs = noSents1;

    // Tokenize sentences
    tokenizeSentences(sentences1, noSents1);
    tokenizeSentences(sentences2, noSents2);

    // Read the features
    readVPSentenceFeatures();
}

// Reading the features
void readVPSentenceFeatures(){
    printf("\nReading other features for the sentences!\n");
    char* cocFeat1 = (char*) malloc(sizeof(char) * 100);
    char* cocFeat2 = (char*) malloc(sizeof(char) * 100);
    char* tfFeat1 = (char*) malloc(sizeof(char) * 100);
    char* tfFeat2 = (char*) malloc(sizeof(char) * 100);
    char* gtPath = (char*) malloc(sizeof(char) * 100);
    char* splitPath = (char*) malloc(sizeof(char) * 100);
    char* validPath = (char*) malloc(sizeof(char) * 100);

    if(debugModeVP){
        // Files for co-occurance features
        cocFeat1 = "/home/satwik/VisualWord2Vec/data/vp_features_coc_1_debug.txt";
        cocFeat2 = "/home/satwik/VisualWord2Vec/data/vp_features_coc_2_debug.txt";

        // Files for total frequency features
        tfFeat1 = "/home/satwik/VisualWord2Vec/data/vp_features_tf_1_debug.txt";
        tfFeat2 = "/home/satwik/VisualWord2Vec/data/vp_features_tf_2_debug.txt";

        // Also read the ground truth file, test/train split, validation set
        gtPath = "/home/satwik/VisualWord2Vec/data/vp_ground_truth_debug.txt";
        splitPath = "/home/satwik/VisualWord2Vec/data/vp_split_debug.txt";
        validPath = "/home/satwik/VisualWord2Vec/data/vp_val_inds_debug.txt";
    }
    else{
        // Files for co-occurance features
        cocFeat1 = "/home/satwik/VisualWord2Vec/data/vp_features_coc_1.txt";
        cocFeat2 = "/home/satwik/VisualWord2Vec/data/vp_features_coc_2.txt";

        // Files for total frequency features
        tfFeat1 = "/home/satwik/VisualWord2Vec/data/vp_features_tf_1.txt";
        tfFeat2 = "/home/satwik/VisualWord2Vec/data/vp_features_tf_2.txt";

        // Also read the ground truth file, test/train split, validation set
        gtPath = "/home/satwik/VisualWord2Vec/data/vp_ground_truth.txt";
        splitPath = "/home/satwik/VisualWord2Vec/data/vp_split.txt";
        validPath = "/home/satwik/VisualWord2Vec/data/vp_val_inds_1k.txt";
        //validPath = "/home/satwik/VisualWord2Vec/data/vp_val_inds.txt";
    }

    FILE* gtFile = fopen(gtPath, "rb");
    FILE* splitFile = fopen(splitPath, "rb");
    FILE* valFile = fopen(validPath, "rb");

    // Read the dimensions and check for match in both the cases
    FILE* cocFile1 = fopen(cocFeat1, "rb");
    FILE* cocFile2 = fopen(cocFeat2, "rb");
    FILE* tfFile1 = fopen(tfFeat1, "rb");
    FILE* tfFile2 = fopen(tfFeat2, "rb");

    // Checking for sanity
    if(cocFile1 == NULL || cocFile2 == NULL || 
        tfFile1 == NULL || tfFile2 == NULL || 
        gtFile == NULL || valFile == NULL){
        printf("\nFiles dont exist to read the features!\n");
        exit(1);
    }

    int featDim11 = 1, featDim21 = 2, featDim12 = 3, featDim22 = 4;
    fscanf(cocFile1, "%d\n", &featDim11);
    fscanf(cocFile2, "%d\n", &featDim12);
    fscanf(tfFile1, "%d\n", &featDim21);
    fscanf(tfFile2, "%d\n", &featDim22);

    // Reading the dimensions
    if(featDim11 != featDim12 || featDim21 != featDim22){
        printf("Feature dimensions dont match !\n(%d, %d), (%d, %d)\n", 
                                    featDim12, featDim11, featDim21, featDim22);
        exit(1);
    }
    else{
        printf("\nFeatures of size : %d %d read!\n", featDim11, featDim22);
    }

    long i, d;
    float feature;
    otherFeatSize = featDim11 + featDim21;
    int totalFeatSize = otherFeatSize + layer1_size;

    // Allocate sufficient memory for each of the features
    for(i = 0; i < noSentPairs; i++){
        // Allocating memory for the features
        sentences1[i].otherFeats = (float*) malloc(sizeof(float) * otherFeatSize);
        sentences2[i].otherFeats = (float*) malloc(sizeof(float) * otherFeatSize);
        
        // Co-occurance feature
        for(d = 0; d < featDim11; d++){
            if(fscanf(cocFile1, "%f", &feature) != EOF)
                sentences1[i].otherFeats[d] = feature;
                
            if(fscanf(cocFile2, "%f", &feature) != EOF)
                sentences2[i].otherFeats[d] = feature;
        }

        // Total frequency feature
        for(d = featDim11; d < otherFeatSize; d++){
            if(fscanf(tfFile1, "%f", &feature) != EOF)
                sentences1[i].otherFeats[d] = feature;
                
            if(fscanf(tfFile2, "%f", &feature) != EOF)
                sentences2[i].otherFeats[d] = feature;
        }
    }
    

    // Compute the features for the sentence pairs
    sentPairs = (struct SentencePair*) malloc(sizeof(struct SentencePair) * noSentPairs);
    long index = 0; 
    int gtruth, isTrain, isVal;
    for(i = 0; i < noSentPairs; i++){
        // Allocating memory for the features
        sentPairs[i].feature = (float*) malloc(sizeof(float) * totalFeatSize * 2);
        
        // Setting the sentence pairs
        sentPairs[i].sent1 = sentences1 + i;
        sentPairs[i].sent2 = sentences2 + i;

        // Haar like maps for the features
        for(d = 0; d < otherFeatSize; d++){
            // Offset leave word2vec features space
            index = d + 2*layer1_size;
            sentPairs[i].feature[index] = sentences1[i].otherFeats[d] + 
                                        sentences2[i].otherFeats[d];
        }

        for(d = 0; d < otherFeatSize; d++){
            // Offset to leave word2vec, otherFeatSize (sum)
            index = d + 2*layer1_size + otherFeatSize;
            sentPairs[i].feature[index] = fabs(sentences1[i].otherFeats[d] -
                                        sentences2[i].otherFeats[d]);
        }

        // Reading the ground truth
        fscanf(gtFile, "%d\n", &gtruth);
        // Check for consistency
        if(!(gtruth == 1 || gtruth == 0)){
            printf("Ground truth unexpected!\n");
            exit(1);
        }
        sentPairs[i].gt = gtruth;

        // Reading if its train / test
        fscanf(splitFile, "%d\n", &isTrain);
        // Check for consistency
        if(!(isTrain == 1 || isTrain == 0)){
            printf("Split unexpected!\n");
            exit(1);
        }
        sentPairs[i].isTrain = isTrain;
        
        // Reading if it is validation set or not
        fscanf(valFile, "%d\n", &isVal);
        // Check for consistency
        if(!(isVal == 1 || isVal == 0)){
            printf("Val set unexpected!\n");
            exit(1);
        }
        sentPairs[i].isVal = isVal;
    }

    // close the file
    fclose(cocFile1);
    fclose(cocFile2);
    fclose(tfFile1);
    fclose(tfFile2);
    fclose(gtFile);
    fclose(splitFile);
    fclose(valFile);
}

// Computing the feaures for the sentences, assume the current embeddings to be updated
void computeSentenceFeatures(){
    printf("\nComputing the features for sentences!\n");
    int d;
    long i;
    
    // Go through all the pairs and compute the embeddings 
    for(i = 0; i < noSentPairs; i++){
        // Get sum features
        for(d = 0; d < layer1_size; d++)
            sentPairs[i].feature[d] = 
                    sentences1[i].embed[d] + sentences2[i].embed[d];

        // Get abs difference features
        for(d = 0; d < layer1_size; d++)
            sentPairs[i].feature[d + layer1_size] = 
                    fabs(sentences1[i].embed[d] - sentences2[i].embed[d]);
    }
}

// Reading all the sentences along with features
// Function to cluster visual features
// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVPAbstractVisualFeatures(int clusters, char* savePath){
    int k = clusters;                           /* number of cluster to create */
    int d = vpFeatSize;                           /* dimensionality of the vectors */
    int n = noTrainVP;                         /* number of vectors */
    //int nt = 1;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */
    
    // Populate the features
    float * v = fvec_new (d * n);    /* random set of vectors */
    long i, j, offset;
    for (i = 0; i < n; i++){
        offset = i * d;
        for(j = 0; j < d; j++)
            v[offset + j] = (float) trainSents[i].vFeat[j];
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
        trainSents[i].cId = assign[i] + 1;
    
    // Debugging the cId for the trainSents tuples
    /*for (i = 0; i < n; i++)
     printf("%i\n", trainSents[i].cId);*/

     // Write the clusters to a file, if non-empty
    if(savePath != NULL){
        // Open the file
        FILE* filePtr = fopen(savePath, "wb");

        if(noTrainVP == 0){
            printf("ClusterIds not available to save!\n");
            exit(1);
        }

        // Save the cluster ids
        int i;
        for (i = 0; i < noTrainVP; i++)
            fprintf(filePtr, "%d %f\n", trainSents[i].cId, dis[i]);

        // Close the file
        fclose(filePtr); 
    }
    
    // Assigning the number of clusters
    if(noClusters == 0){
        noClusters = clusters;
    }
    
    // Free memory
    free(v); free(centroids); free(dis); free(assign); free(nassign);
}

// Refine the network based on the cluster id
void refineNetworkVP(){
    printf("Refining using VP training sentences\n");
    long c, i, s, w;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    int wordCount = 0;
    // The starting and ending index for the current sentence in a description
    int startInd, endInd; 

    // Checking if training examples are present
    if(noTrainVP == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training sentences
    for(i = 0; i < noTrainVP; i++){
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
               
                    printf("Start, end, number: %d %d %d\n", startInd, endInd, endInd - startInd + 1);
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

// Perform the visual paraphrasing task
void performVPTask(){
    // Read the sentence pairs if not read before
    if(noSentPairs == 0)
        readVPSentences();
        
    // Re-compute the embeddings for all the sentences
    computeSentenceEmbeddings(sentences1, noSentPairs);
    computeSentenceEmbeddings(sentences2, noSentPairs);

    // Re-compute features for sentence pairs
    computeSentenceFeatures();

    // Learn the model and return the accuracy
    learnClassificationModel(sentPairs, noSentPairs, (otherFeatSize + layer1_size)*2);
}
