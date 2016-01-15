# include "helperFunctions.h"
// Saving the word2vec vectors for further use
void saveWord2Vec(char* fileName){
    FILE* filePt = fopen(fileName, "wb");

    // Write the vocab size and embedding dimension on the first line
    fprintf(filePt, "%lld %lld\n", vocab_size, layer1_size);

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
    // Have another layer to avoid local variable
    struct Sentence** collection = (struct Sentence**)
                                    malloc(sizeof(struct Sentence*));
    collection[0] = (struct Sentence*) 
                            malloc(sizeof(struct Sentence) * sentCount);
    // Read and store contents
    for( i = 0; i < sentCount; i++){
        if(fgets(currentSent, MAX_SENTENCE, filePtr) != NULL){
            // Remove the trailing \n
            currentSent[strlen(currentSent) - 1] = '\0';
            
            // Allocate memory and copy sentence
            collection[0][i].sent = (char*) malloc(sizeof(char) * MAX_SENTENCE);
            strcpy(collection[0][i].sent, currentSent);
        }

        // Assign other members as needed
        collection[0][i].embed = NULL;

        // Assign some gt
        collection[0][i].gt = -1;
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

    return collection;
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
                collection[i].endIndex[sentCount] = count-1;
                sentCount++;
                lineEnd = 0;
            }
            
            // Next word
            line = strtok(NULL, delim);
        }

        // Punctuations absent, treat everything as one setnence
        if(sentCount == 0){
            sentCount = 1;
            collection[i].endIndex[0] = count-1;
        }

        // Now store the word components, looping over them
        collection[i].count = count;
        collection[i].actCount = actCount;
        collection[i].sentCount = sentCount;

        //printf("Sent count: %s\n%d\n", collection[i].sent, collection[i].sentCount);
    }

    printf("\nTokenized %ld sentences!\n", noSents);
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

/***************************************************/
// Read the visual features (common for all the tasks)
// (parallel)
// Input: 1. Path where features are stored
//        2. placeholder to capture the number of features
//
// Output: 1. Returns the feature (float***)
float*** readVisualFeatures(char* featPath, long* noFeats, int* visualFeatSize){
    // Read the file and setup the threads
    FILE* filePt = fopen(featPath, "rb");

    if(filePt == NULL){
        printf("File at %s doesnt exist!\n", featPath);
        exit(1);
    }

    // Local declarations
    float*** features;
    int i, noLines = 0;
    long noFeatures = 0; // Local variable for noFeats
    int visualFeatureSize = 0; // local variable for visualFeatSize

    // Read the first line and get the feature size
    fscanf(filePt, "%ld %d\n", &noFeatures, &visualFeatureSize);
    printf("\nVisual features are of size: %d...\nNumber of features: %ld ...\n", 
                                visualFeatureSize, noFeatures);
    // Get the initial offset and size of each line
    long offset = ftell(filePt);
    fseek(filePt, 0, SEEK_END);
    long fileSize = ftell(filePt);
    long sizePerLine = (fileSize - offset)/noFeatures;
    // Close the file
    fclose(filePt);

    // Setting up the memory
    features = (float***) malloc(sizeof(float**));
    features[0] = (float**) malloc(sizeof(float*) * noFeatures);
    for (i = 0; i < noFeatures; i++)
        features[0][i] = (float*) malloc(sizeof(float) * visualFeatureSize);

    // Initialize the threads, datastructures
    pthread_t* threads = (pthread_t*) malloc(num_threads * sizeof(pthread_t));
    struct ReadParameter* params = (struct ReadParameter*) 
                            malloc(num_threads * sizeof(struct ReadParameter));

    long startId = 0;
    long endId = noFeatures/num_threads;
    for(i = 0; i < num_threads; i++){
        // create the corresponding datastructures
        params[i].filePath = featPath;
        params[i].features = features;
        params[i].visualFeatSize = visualFeatureSize;
        params[i].threadId = i;
        params[i].startPos = offset + sizePerLine * startId;
        params[i].endPos = offset + sizePerLine * endId;
    
        // start the threads
        if(pthread_create(&threads[i], NULL, readVisualFeaturesThread, &params[i])){
            fprintf(stderr, "error creating thread\n");
            return NULL;
        }
        
        printf("thread: %d (%ld, %ld)  ", i, startId, endId);
        printf("Size: %ld (%ld, %ld) %ld %ld\n\n", params[i].startPos, 
                        params[i].endPos, offset, sizePerLine, fileSize);
        // compute the start and ends for the next thread
        startId = endId; // start from the next one
        if (i != num_threads - 2)
            // add another chunk if not calculating for the last thread
            endId = endId + noFeatures/num_threads;
        else
            // everything till the end for the last thread
            endId = noFeatures;
    }

    // wait for all the threads to finish
    for(i = 0 ; i < num_threads; i++)
        if(pthread_join(threads[i], NULL)){
            fprintf(stderr, "error joining thread\n");
            return NULL;
        }

    printf("\nRead visual features for %d sentences...\n", noLines);
    // Assigning the variables
    *noFeats = noFeatures;
    *visualFeatSize = visualFeatureSize;

    return features;
}

// Thread for reading the visual features
void* readVisualFeaturesThread(void* readParams){
    // Local aliase
    struct ReadParameter* params = readParams;
    float feature;
    float*** features = params->features;
    FILE* filePt = fopen(params->filePath, "r");
    int visualFeatureSize = params->visualFeatSize;
    int i, noLines = 0;

    // Open the file
    // Go to the designated place and start reading until end
    fseek(filePt, params->startPos, SEEK_SET);

    // Reading till designated end is reached
    while(ftell(filePt) <= params->endPos){
        // Read the features
        for(i = 0; i < visualFeatureSize; i++){
            fscanf(filePt, "%f", &feature);
            features[0][noLines][i] = feature;
        }

        // Debugging printing
        if(noLines % 5000 == 0)
            printf("Reading features (%d) : %d\n", params->threadId, noLines);
        noLines++;
    }
    // Closing the file
    fclose(filePt);

    return NULL;
}

/***************************************************/
// Debugging functions
// Write the sentences back to the file to check
void saveSentences(struct Sentence* sents, int noSents, char* savePath){
    // Open the file
    FILE* filePtr = fopen(savePath, "wb");

    if(noSents == 0){
        printf("Sentences not available to save!\n");
        exit(1);
    }

    // Save the cluster ids
    int i;
    for (i = 0; i < noSents; i++)
        fprintf(filePtr, "%s\n", sents[i].sent);
        //fprintf(filePtr, "%d %f\n", assign[i], dis[i]);

    // Close the file
    fclose(filePtr); 
}
