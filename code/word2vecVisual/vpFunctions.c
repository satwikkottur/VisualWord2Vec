// Definitions for functions related to the visual paraphrasing task
# include "vpFunctions.h"

// Globals for the current scope
long noTrainVP = 0;
int vpFeatSize = 0;
int noClustersVP = 0;
struct trainSent* train;

// Read the training sentences for VP task
void readVPTrainSentences(char* featurePath){
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
    
    // Allocate the train
    train = (struct trainSent*) malloc(sizeof(struct trainSent) * sentCount);
    // Read and store contents
    for( i = 0; i < sentCount; i++){
        // Allocating other memories
        //train[i].count = 0;
        //train[i].index = (int*) malloc(0);

        if(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL){
            // Remove the trailing \n
            currentSent[strlen(currentSent) - 1] = '\0';
            
            // Allocate memory and copy sentence
            train[i].sent = (char*) malloc(sizeof(char) * MAX_SENTENCE);
            strcpy(train[i].sent, currentSent);
            
        }
    }

    // Checking with noTrainVP, if exists, else initializing
    if(noTrainVP != 0){
        if(noTrainVP != sentCount){
            printf("Mismatch with number of training examples: %s\n", featurePath);
            exit(1);
        }
    }
    else noTrainVP = sentCount;

    fclose(filePtr);
    printf("\nFile read with %ld sentences!....\n", sentCount);
    
    ////////////////////////////////////////////////////////////////////
    // Writing it back to the file for debugging
    /*FILE* savePtr = fopen("vp_train_emit.txt", "wb");

    for (i = 0; i < noTrainVP; i++){
        fprintf(savePtr, "%s\n", train[i].sent);
    }

    fclose(savePtr);*/
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
    
    int noLines = 0, i;
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
        printf("NoTrainVP: %d\nNo lines: %d\n", noTrainVP, noLines);
        exit(1);
    }
    vpFeatSize = featDim;

    // Closing the file
    fclose(filePt);
}

// Function to tokenize the training sentences and link to word2vec vocab
void tokenizeTrainSentences(){
    long i;
    for (i = 0; i < noTrainVP; i++){
        // Copy the word into a local variable line
        char* line = (char*) malloc(MAX_SENTENCE);
        strcpy(line, train[i].sent);

        int count = 0, n;

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
        train[i].count = count;
        train[i].index = (long*) malloc(train[i].count * sizeof(long));

        line = strtok(temp, delim);
        count = 0;
        while(line != NULL){
            // Convert the token into lower case
            for(n = 0; line[n]; n++) line[n] = tolower(line[n]);
           
            // Save the index
            train[i].index[count] = SearchVocab(line);
            //if(train[i].index[count] == -1)
            //    printf("%s\n", line);
            line = strtok(NULL, delim);
            count++;
        }
    }
}

// FUnction to cluster visual features
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
    if(noClustersVP == 0){
        noClustersVP = clusters;
    }
    
    // Free memory
    free(v); free(centroids); free(dis); free(assign); free(nassign);
}

// Refine the network based on the cluster id
