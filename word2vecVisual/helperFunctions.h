# ifndef HELPER_FUNCS
# define HELPER_FUNCS

// Standard libraries
# include <stdio.h>
# include <stdlib.h>
# include <string.h>
# include <math.h>
# include <pthread.h>
# include <ctype.h>

// For clustering
# include <yael/vector.h>
# include <yael/kmeans.h>
# include <yael/machinedeps.h>
# include "structs.h"
# include "macros.h"

// Declaring the extern variables allowing separation of code
extern long long vocab_size, layer1_size;
extern float *syn0, *syn1, *expTable;
extern struct vocab_word* vocab;
// Returns position of a word in the vocabulary; if the word is not found, returns -1
int SearchVocab(char *word);

// Saving the word2vec vectors for further use
void saveWord2Vec(char*);
// Load the word2vec vectors
// We assume all the other parameters(vocabulary is kept constant)
// Use with caution
void loadWord2Vec(char*);

// Multiple character split
// Source: http://stackoverflow.com/questions/29788983/split-char-string-with-multi-character-delimiter-in-c
char *multi_tok(char *, char *);

/***************************************************/
// Read the sentences
struct Sentence** readSentences(char*, long*);

// Tokenize sentences
void tokenizeSentences(struct Sentence*, long);

// Save the sentence embeddings
void writeSentenceEmbeddings(char*, struct Sentence*, long);


/***************************************************/

# endif
