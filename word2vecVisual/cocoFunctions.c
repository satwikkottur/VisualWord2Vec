# include "cocoFunctions.h"

// Variables for the current task
static struct Sentence* trainSents;
static float** features;
static int* featClusterId = NULL;
static long noTrain = 0;
static long noFeats = 0;

// Reading the  training sentences
void readTrainSentencesCOCO(char* trainPath, char* mapPath){
    long noSents = 0;
    // Use readSentences
    trainSents = *readSentences(trainPath, &noSents);

    if(noTrain != 0){
        if(noTrain != noSents){
            printf("Mismatch with number of training examples: %s\n", trainPath);
            exit(1);
        }
    }
    else noTrain = noSents;

    // Now reading the maps
    FILE* mapPtr = fopen(mapPath, "rb");

    if(mapPtr == NULL){
        printf("File doesnt exist at %s!\n", mapPath);
        exit(1);
    }

    int i, mapId;
    for(i = 0; i < noTrain; i++)
        if(fscanf(mapPtr, "%d\n", &mapId) != EOF)
            trainSents[i].featInd = mapId;

    fclose(mapPtr);
    printf("\nRead %ld sentences for training!\n", noTrain);

    // Debug, checking the feature indices
    /*for(i = 0; i < noTrain; i++){
        printf("Train map index: %d => %d\n", i, trainSents[i].featInd);
    }
    exit(1);*/
}

// Reading the cluster ids
void readClusterIdCOCO(char* clusterPath){
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
void tokenizeTrainSentencesCOCO(){
    // Call the function to tokenize sentences
    tokenizeSentences(trainSents, noTrain);
}

// Refining the network using COCO
void refineNetworkCOCO(){
    printf("\nRefining using MSCOCO training sentences....\n");
    refineNetworkSentences(trainSents, noTrain, trainMode);
}

// Reading the visual feature file for COCO
void readVisualFeatureFileCOCO(char* featPath){
    FILE* filePt = fopen(featPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", featPath);
        exit(1);
    }

    float feature;
    int i, noLines = 0;

    // Read the first line and get the feature size
    fscanf(filePt, "%ld %d", &noFeats, &visualFeatSize);
    printf("\nVisual features are of size: %d...\nNumber of features: %ld ...\n", 
                                visualFeatSize, noFeats);

    // Setting up the memory
    features = (float**) malloc(sizeof(float*) * noFeats);
    for (i = 0; i < noFeats; i++)
        features[i] = (float*) malloc(sizeof(float) * visualFeatSize);

    // Reading till EOF
    while(fscanf(filePt, "%f", &feature) != EOF){
        // Save the already read feature
        features[noLines][0] = feature;

        for(i = 1; i < visualFeatSize; i++){
            fscanf(filePt, "%f", &feature);
            features[noLines][i] = feature;
        }

        // Debugging printing
        if(noLines % 5000 == 0)
            printf("Reading features : %d\n", noLines);

        noLines++;
    }

    if(noLines != noFeats){
        printf("Number of features incorrectly read!\n");
        exit(1);
    }

    printf("\nRead visual features for %d sentences...\n", noLines);

    // Closing the file
    fclose(filePt);
}

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeaturesCOCO(int clusters, char* savePath){
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
