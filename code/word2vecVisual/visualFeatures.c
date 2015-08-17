# include "functionSigns.h"
# include "visualFeatures.h"

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
        // Getting the indices for p, s, r
        struct prsTuple newTuple;/* = {.p = {findTupleIndex(pWord)},
                                    .r = {findTupleIndex(rWord)},
                                    .s = {findTupleIndex(sWord)},
                                    .cId = -1,
                                    .feat = NULL,
                                    .embed = NULL};*/
        //printf("%s : %s : %s\n", p, s, r);
        newTuple.p = findTupleIndex(pWord);
        newTuple.s = findTupleIndex(sWord);
        newTuple.r = findTupleIndex(rWord);

        prs[noTuples] = newTuple;
        noTuples++;
    }

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
        if(prs[i].cId == -1) prs[i].cId = clusterId;
        i++;
    }

    // Sanity check
    if(i != NUM_TRAINING){
        printf("\nNumber of training instances dont match in cluster file!\n");
        exit(1);
    }

    fclose(filePt);
}

// Finding the indices of words for P,R,S
struct featureWord findTupleIndex(char* word){
    int index = SearchVocab(word); 

    struct featureWord feature = {.str = word};
    
    // Do something if not in vocab
    if(index == -1) {
        //printf("Not in vocab -> %s : %s\n", word, "") ;

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
        int count = 0;
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
            int i;
            for(i = 0; token[i]; i++) token[i] = tolower(token[i]);
           
            // Save the index
            feature.index[count] = SearchVocab(token);
            if(feature.index[count] == -1)
                printf("Word not found in dictionary => %s\t |  %s\n", token, word);

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
