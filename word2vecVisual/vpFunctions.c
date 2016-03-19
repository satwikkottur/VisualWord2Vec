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
}

// Saving word2vec embeddings for all VP sentences
void writeVPSentenceEmbeddings(){
    // Path to the sentences_1
    char readSent1[] = "/home/satwik/VisualWord2Vec/data/vp/vp_sentences1_lemma.txt";
    char writeSent1[] = "/home/satwik/VisualWord2Vec/data/vp/vp_30_features_1.txt";
    //char writeSent1[] = "/home/satwik/VisualWord2Vec/data/vp_orig_features_1.txt";
    // Path to the sentences_2
    char readSent2[] = "/home/satwik/VisualWord2Vec/data/vp/vp_sentences2_lemma.txt";
    char writeSent2[] = "/home/satwik/VisualWord2Vec/data/vp/vp_30_features_2.txt";
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
    long noSents1, noSents2;
    sentences1 = *readSentences(VP_TASK_SENTENCES_1, &noSents1);
    sentences2 = *readSentences(VP_TASK_SENTENCES_2, &noSents2);
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

    FILE* gtFile = fopen(VP_GROUND_TRUTH_FILE, "rb");
    FILE* splitFile = fopen(VP_TEST_TRAIN_SPLIT, "rb");
    FILE* valFile = fopen(VP_VAL_SPLIT, "rb");

    // Read the dimensions and check for match in both the cases
    FILE* cocFile1 = fopen(VP_CO_OCCUR_1, "rb");
    FILE* cocFile2 = fopen(VP_CO_OCCUR_2, "rb");
    FILE* tfFile1 = fopen(VP_TOTAL_FREQ_1, "rb");
    FILE* tfFile2 = fopen(VP_TOTAL_FREQ_2, "rb");

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

// Refine the network based on the cluster id, given sentences
void refineNetworkVP(){
    printf("\nRefining using VP training sentences....\n");
    refineNetworkSentences(trainSents, noTrainVP, trainMode);
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
