# include "genomeFunctions.h"

// Variables for the current task
static struct Sentence* trainSents;
static float** features;
static int* featClusterId = NULL;
static long noTrain = 0;
static long noFeats = 0;

// Reading the  training sentences, ignore the map
void readTrainSentencesGenome(char* trainPath){
    long noSents = 0;
    // Use readSentences
    trainSents = *readSentences(trainPath, &noSents);

    // Double check if read before
    if(noTrain != 0){
        if(noTrain != noSents){
            printf("Mismatch with number of training examples: %s\n", trainPath);
            exit(1);
        }
    }
    else noTrain = noSents;

    int i;
    for(i = 0; i < noTrain; i++)
        trainSents[i].featInd = i;

    printf("\nRead %ld sentences for training!\n", noTrain);

    // Debug, checking the feature indices
    /*for(i = 0; i < noTrain; i++){
        printf("Train map index: %d => %d\n", i, trainSents[i].featInd);
    }
    exit(1);*/
    // Debug, re write the sentences back to cross check
    // Write the sentences back to check debug
    /*char* writePath = (char*) malloc(sizeof(char) * 100);
    writePath = "/home/satwik/VisualWord2Vec/data/vis-genome/train/re-written.txt";
    saveSentences(trainSents, noTrain, writePath);*/
}

// Reading the cluster ids
void readClusterIdGenome(char* clusterPath){
    FILE* filePt = fopen(clusterPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", clusterPath);
        exit(1);
    }

    // Check if the feature cluster ids are initialized
    if(noFeats == 0){
        printf("Features not read!\n");
        exit(1);
    }
    else if (featClusterId == NULL)
        featClusterId = (int*) malloc(sizeof(int) * noFeats);

    // Keep track of max clsuter id
    int i = 0, clusterId, maxClusterId = 0;
    while(fscanf(filePt, "%d\n", &clusterId) != EOF){
        featClusterId[i] = clusterId + 1;
        i++;
        if(maxClusterId < clusterId) maxClusterId = clusterId;
    }

    // Sanity check
    if(i != noFeats){
        printf("\nNumber of features dont match in cluster file!\n");
        exit(1);
    }
    else{
        printf("\nRead cluster file with K = %d\n", maxClusterId+1);
        noClusters = maxClusterId + 1;
    }

    // Assigning the cluster ids to the sentences
    for(i = 0; i < noTrain; i++){
        trainSents[i].cId = featClusterId[trainSents[i].featInd];
    }

    fclose(filePt);
}

// Tokenize the sentences for training
void tokenizeTrainSentencesGenome(){
    // Call the function to tokenize sentences
    tokenizeSentences(trainSents, noTrain);
}

// Refining the network using VQA
void refineNetworkGenome(){
    printf("\nRefining using Visual Genome training sentences....\n");
    refineNetworkSentences(trainSents, noTrain, trainMode);
}

// Reading the visual feature file for VQA
void readVisualFeatureFileGenome(char* featPath){
    // Read the features for the genome dataset
    features = *readVisualFeatures(featPath, &noFeats, &visualFeatSize);
}

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeaturesGenome(int clusters, char* savePath){
    int k = clusters;                           /* number of cluster to create */
    int d = visualFeatSize;                           /* dimensionality of the vectors */
    int n = noFeats;                         /* number of vectors */
    int nt = 12;                           /* number of threads to use */
    int niter = 0;                        /* number of iterations (0 for convergence)*/
    int redo = 1;                         /* number of redo */
    
    // Populate the features
    float * v = fvec_new (d * n);    /* random set of vectors */
    long i, j, offset;
    for (i = 0; i < n; i++){
        offset = i * d;
        for(j = 0; j < d; j++)
            v[offset + j] = (float) features[i][j];
    }
    
    /* variables are allocated externaly */
    float * centroids = fvec_new (d * k); /* output: centroids */
    float * dis = fvec_new (n);           /* point-to-cluster distance */
    int * assign = ivec_new (n);          /* quantization index of each point */
    int * nassign = ivec_new (k);         /* output: number of vectors assigned to each centroid */
    
    double t1 = getmillisecs();
    // Cluster the features
    kmeans (d, n, k, niter, v, nt, 1, redo, centroids, dis, assign, nassign);
    double t2 = getmillisecs();
    
    printf ("kmeans performed in %.3fs\n\n", (t2 - t1)  / 1000);
    //ivec_print (nassign, k);
    
    // Write the cluster ids to the Sentences, considering their featId
    featClusterId = (int*) malloc(sizeof(int) * noFeats);
    for (i = 0; i < n; i++)
        featClusterId[i] = assign[i] + 1;

    for (i = 0; i < noTrain; i++)
        trainSents[i].cId = assign[trainSents[i].featInd] + 1;

    // Debugging the cId for the train tuples
    /*for (i = 0; i < n; i++)
     printf("%i\n", train[i].cId);*/

     //==================================================================
     // Write the clusters to a file, if non-empty
    if(savePath != NULL){
        // Open the file
        FILE* filePtr = fopen(savePath, "wb");

        if(noTrain == 0){
            printf("ClusterIds not available to save!\n");
            exit(1);
        }

        // Save the cluster ids
        for (i = 0; i < n; i++)
            fprintf(filePtr, "%d\n", assign[i]);
            //fprintf(filePtr, "%d %f\n", assign[i], dis[i]);

        // Close the file
        fclose(filePtr); 
    }
    
    // Assigning the number of clusters
    if(noClusters == 0) noClusters = clusters;
    
    // Free memory
    free(v); free(centroids); free(dis); free(assign); free(nassign);
}

