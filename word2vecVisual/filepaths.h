# ifndef PATH_FILE
# define PATH_FILE

// NOTE: 
//
// Re-compile if any of the variables are changed below
// as it is included as header file
// To do it:
// make clean; make
//

// Common sense assertion : CS 
// Path to the visual features, header = dimension of the feature
# define CS_VISUAL_FEATURE_FILE "/home/satwik/VisualWord2Vec/data/float_features.txt"
// Path to PRS file for TRAIN tuples, format:<P:S:R>
# define CS_PRS_TRAIN_FILE "/home/satwik/VisualWord2Vec/data/PSR_features.txt"
// Path to PRS file for TEST tuples, format:<P:R:S> gt
# define CS_PRS_TEST_FILE "/home/satwik/VisualWord2Vec/data/test_features.txt"
// Path to PRS file for VAL tuples, format:<P:R:S> gt
# define CS_PRS_VAL_FILE "/home/satwik/VisualWord2Vec/data/val_features.txt"

#endif

