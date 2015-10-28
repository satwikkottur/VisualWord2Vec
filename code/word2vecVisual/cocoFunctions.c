# include "cocoFunctions.h"

// Variables for the current task
static struct Sentence* trainSents;
static long noTrain = 0;
static int featSize = 0;

// Reading the  training sentences
void readTrainSentencesCOCO(char* trainPath){
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
    printf("\nRead %ld sentences for training!\n", noTrain);
}

// Reading the cluster ids
void readClusterIdCOCO(char* clusterPath){
    FILE* filePt = fopen(clusterPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", clusterPath);
        exit(1);
    }

    // Keep track of max clsuter id
    int i = 0, clusterId, maxClusterId = 0;
    while(fscanf(filePt, "%d\n", &clusterId) != EOF){
        trainSents[i].cId = clusterId + 1;
        i++;
        if(maxClusterId < clusterId) maxClusterId = clusterId;
    }

    // Sanity check
    if(i != noTrain){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }
    else{
        printf("\nRead cluster file with K = %d\n", maxClusterId+1);
        noClusters = maxClusterId + 1;
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
    fscanf(filePt, "# %d", &visualFeatSize);
    printf("Visual features are of size : %d...\n", visualFeatSize);

    // Reading till EOF
    while(fscanf(filePt, "%f", &feature) != EOF){
        trainSents[noLines].vFeat = (float*) malloc(sizeof(float) * visualFeatSize);
        // Save the already read feature
        trainSents[noLines].vFeat[0] = feature;

        for(i = 1; i < visualFeatSize; i++){
            //printf("%f ", feature);
            fscanf(filePt, "%f", &feature);
            trainSents[noLines].vFeat[i] = feature;
        }
        //printf("%f\n", feature);

        if(noLines % 5000 == 0)
            printf("Line : %d\n", noLines);
        noLines++;
    }

    printf("\nRead visual features for %d sentences...\n", noLines);
    featSize = noLines;

    // Closing the file
    fclose(filePt);
}

// Clustering kmeans wrapper
// Source: http://yael.gforge.inria.fr/tutorial/tuto_kmeans.html
void clusterVisualFeaturesCOCO(int clusters, char* savePath){
    int k = clusters;                           /* number of cluster to create */
    int d = visualFeatSize;                           /* dimensionality of the vectors */
    int n = featSize;                         /* number of vectors */
    int nt = 12;                           /* number of threads to use */
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
    kmeans (d, n, k, niter, v, nt, 1, redo, centroids, dis, assign, nassign);
    double t2 = getmillisecs();
    
    printf ("kmeans performed in %.3fs\n\n", (t2 - t1)  / 1000);
    //ivec_print (nassign, k);
    
    // Write the cluster ids to the Sentences
    for (i = 0; i < n; i++)
        trainSents[i].cId = assign[i] + 1;
    
    // Debugging the cId for the train tuples
    /*for (i = 0; i < n; i++)
     printf("%i\n", train[i].cId);*/

     // Write the clusters to a file, if non-empty
    if(savePath != NULL){
        // Open the file
        FILE* filePtr = fopen(savePath, "wb");

        if(noTrain == 0){
            printf("ClusterIds not available to save!\n");
            exit(1);
        }

        // Save the cluster ids
        int i;
        for (i = 0; i < n; i++)
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
