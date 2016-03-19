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
 
// Visual Paraphrase : VP
// Path to the visual feature file
#define VP_VISUAL_FEATURE_FILE "/home/satwik/VisualWord2Vec/data/vp/abstract_features_train.txt"
// Path to text
#define VP_TRAIN_CAPTION_FILE "/home/satwik/VisualWord2Vec/data/vp/vp_train_full.txt"
// Reading the VP task related files
#define VP_TASK_SENTENCES_1 "/home/satwik/VisualWord2Vec/data/vp/vp_sentences1_lemma.txt"
#define VP_TASK_SENTENCES_2 "/home/satwik/VisualWord2Vec/data/vp/vp_sentences2_lemma.txt"
// Path to files that hold features extracted from Xiao's code
// Co-occurance features, total frequency features along with ground truth, test/train split
// and validation set
// Files for total frequency features
# define VP_CO_OCCUR_1 "/home/satwik/VisualWord2Vec/data/vp/vp_features_coc_1.txt";
# define VP_CO_OCCUR_2 "/home/satwik/VisualWord2Vec/data/vp/vp_features_coc_2.txt";

// Files for total frequency features
# define VP_TOTAL_FREQ_1 "/home/satwik/VisualWord2Vec/data/vp/vp_features_tf_1.txt";
# define VP_TOTAL_FREQ_2 "/home/satwik/VisualWord2Vec/data/vp/vp_features_tf_2.txt";

// Ground truth, test/train split, val/train split
# define VP_GROUND_TRUTH_FILE "/home/satwik/VisualWord2Vec/data/vp/vp_ground_truth.txt";
# define VP_TEST_TRAIN_SPLIT "/home/satwik/VisualWord2Vec/data/vp/vp_split.txt";
# define VP_VAL_SPLIT "/home/satwik/VisualWord2Vec/data/vp/vp_val_inds_1k.txt";
#endif
