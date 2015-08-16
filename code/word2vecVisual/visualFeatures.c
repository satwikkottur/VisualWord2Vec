# include "visualFeatures.h"

void readFeatureFile(char* filepath){
    printf("Reading the feature file : %s\n\n", filepath);
    
    // Opening the file
    FILE* filePt = fopen(filepath, "rb");
    printf("Reading %s...\n", filepath);

    char p[MAX_STRING_LENGTH], s[MAX_STRING_LENGTH], r[MAX_STRING_LENGTH];

    // Read and store the contents
    int noTuples = 0;
    while(fscanf(filePt, "<%[^<>:]:%[^<>:]:%[^<>:]>\n", p, s, r) != EOF){
        struct prsTuple newTuple = {.p = p, .r = r, .s = s, .feature = NULL};
        //printf("%s : %s : %s\n", p, s, r);
        
        prs[noTuples] = newTuple;
        noTuples++;
    }

    printf("File read with %d tuples\n", noTuples);
}
