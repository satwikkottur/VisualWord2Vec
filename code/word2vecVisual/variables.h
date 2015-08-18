#ifndef ORIGINAL_HEADER
#define ORIGINAL_HEADER

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;

// Globals
extern int binary, cbow, debug_mode, window, min_count, num_threads, min_reduce;

extern long long vocab_max_size, vocab_size, layer1_size;
extern long long train_words, word_count_actual, iter, file_size, classes;
extern real alpha, starting_alpha, sample;
extern real *syn0, *syn1, *syn1neg, *expTable;

extern const int vocab_hash_size;  // Maximum 30 * 0.7 = 21M words in the vocabulary

extern int hs, negative;
extern const int table_size;

#endif
