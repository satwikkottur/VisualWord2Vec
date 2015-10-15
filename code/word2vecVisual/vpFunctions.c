// Definitions for functions related to the visual paraphrasing task
# include "vpFunctions.h"

// Globals for the current scope
long noTrainVP = 0;
int vpFeatSize = 0;
struct Sentence* train;

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
        // Copy the word into a local variable line
        char* line = (char*) malloc(MAX_SENTENCE);
        strcpy(line, collection[i].sent);

        int count = 0, n, actCount = 0;

        // Split based on 's
        char* first = multi_tok(line, "'s");
        char* second = multi_tok(NULL, "'s");

        // Join both the parts without the 's (from baseline: add it at the end)
        if(second != NULL) line = strcat(first, strcat(second, " \'s"));
        else line = first;

        char* temp = (char*) malloc(MAX_SENTENCE);
        strcpy(temp, line);
        
        // Remove ' ', ',', '.', '?', '!', '\', '/'
        char* delim = " .,/!?\\";
        line = strtok(line, delim);
        // Going over the line to determine the number of parts
        while(line != NULL){
            count++;
            line = strtok(NULL, delim);
        }

        // Now store the word components, looping over them
        collection[i].index = (long*) malloc(count * sizeof(long));

        line = strtok(temp, delim);
        count = 0;
        while(line != NULL){
            // Convert the token into lower case
            for(n = 0; line[n]; n++) line[n] = tolower(line[n]);
           
            // Save the index
            collection[i].index[count] = SearchVocab(line);
            if(collection[i].index[count] != -1) actCount++;
            //if(collection[i].index[count] == -1)
            //    printf("%s\n", line);
            line = strtok(NULL, delim);
            count++;
        }
        // Now store the word components, looping over them
        collection[i].count = count;
        collection[i].actCount = actCount;
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
                mean[d] /= collection[i].actCount;
        
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
    train = *readSentences(featurePath, &noSents);

    if(noTrainVP != 0){
        if(noTrainVP != noSents){
            printf("Mismatch with number of training examples: %s\n", featurePath);
            exit(1);
        }
    }
    else noTrainVP = noSents;
}

// Read the visual features for VP task
void readVPVisualFeatures(char* visualPath){
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
        train[noLines].vFeat = (float*) malloc(sizeof(float) * featDim);
       
        // First entry
        train[noLines].vFeat[0] = feature;

        for (i = 1; i < featDim; i++){
            fscanf(filePt, "%f", &feature);
            train[noLines].vFeat[i] = feature;
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
    tokenizeSentences(train, noTrainVP);

    // Debug
    /*long i, j;
    for (i = 0; i < noTrainVP; i++){
        printf("%s\n", train[i].sent);
        for (j = 0; j < train[i].count; j++){
            printf("%ld ", train[i].index[j]);
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

// Function to cluster visual features
// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVPVisualFeatures(int clusters, char* savePath){
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
            v[offset + j] = (float) train[i].vFeat[j];
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
            fprintf(filePtr, "%d %f\n", train[i].cId, dis[i]);

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
    long c, i;
    float* y = (float*) malloc(sizeof(float) * noClusters);
    int* wordList = (int*) malloc(MAX_SENTENCE * sizeof(int));
    int wordCount = 0;

    // Checking if training examples are present
    if(noTrainVP == 0){
        printf("Training examples not loaded!\n");   
        exit(1);
    }

    // Read each of the training sentences
    for(i = 0; i < noTrainVP; i++){
        // printf("Training %ld instance ....\n", i);
        
        // Checking possible fields to avoid segmentation error
        if(train[i].cId < 1 || train[i].cId > noClusters) {
            printf("\nCluster id (%d) for %ld instance invalid!\n", train[i].cId, i);
            exit(1);
        }

        // Now collecting words for training
        wordCount = 0;
        
        for(c = 0; c < train[i].count; c++){
            // If not in vocab, continue
            if(train[i].index[c] == -1) continue;

            wordList[wordCount] = train[i].index[c];
            // Getting the actual count of words
            wordCount++;
        }
        // Predict the cluster
        computeMultinomialPhrase(y, wordList, wordCount);
        
        // Propage the error the embeddings
        updateWeightsPhrase(y, wordList, wordCount, train[i].cId);
    }
}

