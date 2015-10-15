#ifndef STRUCTS
#define STRUCTS

// Structure to hold the index information
struct featureWord{
    char* str;
    int count;
    int* index;
    // Word embedding for the instance
    float* embed; 
    float magnitude;
    // Word embeddings for each of the model
    float* embedR, *embedP, *embedS;
    float magnitudeR, magnitudeP, magnitudeS;
};

// Structure to hold information about P,R,S triplets
struct prsTuple{
    int p, r, s;
    //struct featureWord p, r, s;
    
    // visual features for the instance 
    float* feat;
    // Cluster id assigned to the current instance
    int cId; 
};

// Structure
struct vocab_word {
  long long cn;
  int *point;
  char *word, *code, codelen;
};

// Structure to hold information about vp task
struct trainSent{
    // Actual sentences
    char* sent;

    // no of words recognized in the current vocab
    int count;

    // indices of the words recognized in the curent vocab
    long* index;

    // Training visual feature used for clustering
    float* vFeat;
};

#endif
