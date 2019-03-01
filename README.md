# Visual Word2Vec (vis-w2v)
Learning visually grounded word embeddings from abstract image  

----

###Paper
**Satwik Kottur\*, Ramakrishna Vedantam\*, Jos&eacute; Moura, Devi Parikh**  
*Visual Word2Vec (vis-w2v): Learning Visually grounded embeddings from abstract images*  
[[ArXiv](http://arxiv.org/abs/1511.07067)] [[Project Page](https://satwikkottur.github.io/research/vis-w2v/)]  
\* = equal contribution

Pre-trained Visual Word2vec Embeddings can be found [here](https://computing.ece.vt.edu/~satwik/visual_word2vec/),
trained on wikipedia and MSCOCO captions.

----
### Code Structure
The code is organized as follows:

* `visword2vec.c` : Main code for training vis-w2v
* `refineFunctions.c` : Contains functions related to refining embeddings
* `helperFunctions.c` : Mostly contain assisting functions like io, tokenization, etc.
* `visualFunction.c` : Contains code to refine based on tuples and also perform common sense (cs) task on test and validation sets
* `vpFunctions.c` : Contains code to refine based on sentences and also perform visual paraphrasing (vp) task
* `structs.h` : Contains structures defined for the code
* `filepaths.h` : Contains the paths to the files needed for the above two tasks-cs,vp
* `Makefile` : Helps to setup, compile and run programs

Other files can be ignored for now.  

Program accepts following as inline arguments:  

1. `embed-path` : Initialization for the embeddings (pre-trained using word2vec usually)  
  Format: Header should have `vocabsize` `dimensions`  
  Each following row should first have the word, and embeddings delimited by space  
1. `output` : Path to where to store the output embeddings
1. `size` : Size of the hidden layer (should match with the pre-loaded embeddings)
1. `threads` : Number of threads to use for refining, loading and other operations

Currently only saving one embedding is supported. For multi embeddings simply turn
the `trainMulti` flag (top of `visword2vec.c` file) to 1. However, saving can be done by
uncommenting code in `trainModel()`.

#### Steps for usage

1. `Makefile`
    * Liblinear-2.1 must be compiled and the path must be correctly set
    * yael must be setup (for k means) and corresponding paths setup
    Link here: [yael](https://gforge.inria.fr/projects/yael/)
    * cs and vp options should have correctly `-embed-path` options

1. `filepaths.h`:
    * Make sure all the paths are accessible and correctly set
    * Any change to this file, should be followed by re-compiling the code

1. `liblinearWrapper.h`:
    * Additionally, you also need to link the correct path to liblibear

1. Other dependencies:
    * NTLK is used for tokenization and lemmatization (VP task)

To run either cs or vp, comment or uncomment corresponding wrapper calls in 
`trainModel()` function of visword2vec. And then `make cs` or `make vp` for the
two tasks correspondingly to compile and run. `make` simply compiles while 
`make clean` cleans up all the binaries.  

**NOTE**: All the binaries are stored in `bin/` folder (might have to create one if 
doesnt exist beforehand).  

----
### Tasks
In this paper, we deal with three tasks: Common Sense Assertion Classification, Visual Paraphrasing and Text-based Image Retrieval.

**A. Common Sense Assertion Classification** ([Project page](https://vision.ece.vt.edu/cs/))  
Download the dataset from their project page [here](https://vision.ece.vt.edu/cs/cs_code_data.zip).
Code to process this dataset further is given in `utils/cs/`.
The following are the pre-processing steps:

1. Extract the training (P, R, S) tuples
2. Extract the visual features for clustering
2. Extract the test and val (P, R, S) tuples
3. Extract the `word2vec` embeddings to initialize from (you can alternatively use any other embeddings to begin with, we recommend you use pre-trained embeddings to reproduce results from the paper).

All the above four steps can be done by simply running:
```
cd utils/cs/
python extractData.py <path to downloaded data> <path to save the data>(optional)
```
By default it created a folder `data/cs` and saves the files in this folder. This will produce files `word2vec_cs.bin`, `PRS_train.txt`, `PRS_test.txt`, `PRS_val.txt` and `visual_train.txt` at destination folder corresponding to above files. Once these files are produced, open `filepath.h` and make sure the macros point to right file paths.
```
# define ROOT_CS "data/cs/"
 
# define CS_VISUAL_FEATURE_FILE ROOT_CS  "visual_train.txt"
# define CS_PRS_TRAIN_FILE ROOT_CS "PRS_train.txt"
# define CS_PRS_TEST_FILE ROOT_CS "PRS_test.txt"
# define CS_PRS_VAL_FILE ROOT_CS "PRS_val.txt"
```
Now, to run, simply:
```
make
./visword2vec -cs 1 -embed-path data/cs/word2vec_cs.bin -output cs_refined.bin -size 200 -clusters 25
```
You can also give in other parameters to suit your needs.

----

**B. Visual Paraphrasing** ([Project page](https://filebox.ece.vt.edu/~linxiao/imagine/))  
Download the VP dataset from their project page [here](https://filebox.ece.vt.edu/~linxiao/imagine/site_data/imagine_v1.zip).
Also download the clipart scenes and descriptions (ASD) used to train `vis-w2v` from the [clipart](https://vision.ece.vt.edu/clipart/) project page [here](http://research.microsoft.com/research/downloads/details/73537628-df14-44e2-847a-45f369131e87/details.aspx).

All the scripts needed for pre-processing are available in `utils/vp` folder. 

Follow the steps below:
**Training data** 
**Step 1:** Run the `fetchVPTrainData.m` function to extract relevant data for training `vis-w2v`.
```
cd utils/vp
>> fetchVPTrainData(<path to ASD dataset>, <path to VP dataset>, <path to save the data>);

For example:
>> fetchVPTrainData('data/vp/AbstractScenes_v1.1', 'data/vp/imagine_v1/', 'data/vp/');
```

It does the following (not important. If you just want desired data, run the above command):
  * Extracting visual features `abstract_features.txt` from Abstract Scene Dataset (ASD) using MATLAB script.
  ```
  >> cd utils/vp
  >> extractAbstractFeatures(<path to ASD dataset>, <path to save the data>)
  
  For example: 
  >> extractAbstractFeatures('data/vp/AbstractScenes_v1.1', 'data/vp/')
  ```
  * The alignment between ASD and VP datasets is given in two files `SceneMap.txt` and `SceneMapV1_10020.txt` present in `utils/vp/`. We will use them along with train/test split of VP and select features from training sentences only, again using MATLAB.
  ```
  cd utils/vp
  >> alignAbstractFeatures(<path to VP dataset>, <path to abstract_features.txt>, <path to save the data>)
  
  For example:
  >> alignAbstractFeatures('data/vp/imagine_v1/', 'data/vp/', 'data/vp/')
  ```
  * Get the training sentences from VP dataset (for learning `vis-w2v`). This would produce `vp_train_sentences_raw.txt`.
  ```
  cd utils/vp
  >> saveVPTrainSentences(<path to VP dataset>, <path to save sentences>)
  
  For example:
  >> saveVPTrainSentences('data/vp/imagine_v1/', 'data/vp')
  ```

**Task data**  
**Step 2:** One should use our new embeddings `vis-w2v` in place of `word2vec` in visual paraphrasing task (`imagine_v1/code/feature/compute_features_vp.m` at line 32). Alternatively, we can save their other text features (co-occurance and total frequency) and use it in our code for speed and smoother interface between learning embeddings and performing the task. 
This can be achieved by adding the following lines to `imagine_v1/code/feature/compute_features_vp.m` before line 30, and running `imagine_v1/code/script_vp.m`.

```
save('vp_txt_features.mat', 'feat_vp_text_tf_1', 'feat_vp_text_tf_2', 'feat_vp_text_coc_1', 'feat_vp_text_coc_2');
% Escape from running remainder code
error('Saved features, getting out!')
```

**Step 3**: Next, we obtain all the relevant information to perform the visual paraphrasing task (using MATLAB)
* Sentence pairs: `vp_sentences_1.txt` and `vp_sentences_2.txt`
* Other textual features: `vp_features_coc_l.txt`, `vp_features_coc_2.txt`, `vp_features_tf_l.txt`, `vp_features_tf_2.txt`
* Ground truth: `vp_ground_truth.txt`
* Train / test split: `vp_split.txt`
* Train / val split: `vp_val_inds_1k.txt`

```
cd /utils/vp
>> fetchVPTaskData(<path to VP dataset>, <path to vp_txt_features.mat>, <path to save the files>)

For example:
>> fetchVPTaskData('data/vp/imagine_v1/', 'data/vp/imagine_v1/code/', 'data/vp')
```

**Step 4**: Finally, we tokenize and lemmatize all the sentences, i.e, `vp_sentences_1.txt`, `vp_sentences_2.txt` and `vp_train_sentences_raw.txt`.

```
cd /utils/vp
python lemmatizeVPTrain.py <path to data> <path to save sentences>

For example:
python lemmatizeVPTrain.py data/vp/ data/vp/
```
Phew! That's a lot of pre-processing. Now we are all set to learn vis-w2v embeddings while performing the visual paraphrasing task.
Like before, check all the filepaths in `filepaths.h` before proceeding.

Now, to run, simply:
```
make
./visword2vec -vp 1 -embed-path data/vp/word2vec_vp.bin -output vp_refined.bin -size 200 -clusters 100
```
You can also give in other parameters to suit your needs. 
For VP, these are `mode` that indicates the training context and `window-size`  that indicates the size in `WINDOW` mode.
  * `DESCRIPTIONS`: Use all the three sentences for training
  * `SENTENCES`: Use sentences one after the other
  * `WINDOW`: Use a context window of size `window-size` (default 5)
  * `WORDS`: Use each word separately

The program prints 100 runs with both validation and test performance. We choose the run with best validation performance and report the corresponding test result.

----

**C. Text-based Image Retrieval**  
This task involves retrieving the appropriate image based on the associated tuple. We collect the data and make it available at `utils/text-ret/text_ret_tuples.pickle` along with ground truths for each image as `utils/text-ret/text_ret_gt.txt`. The goal is to retrieve correct image from the list of ground truth tuples using each of collected queries in pickle file as query.

The code for this task is provided in Python in `utils/text_ret/` along with the data. To run, we need to point it to the data directory along with embedding paths. There are two modes for this task: (A) SINGLE - this uses single embedding for P, R, S. (B) MULTI - this uses three different embeddings for each of P, R, S. The inputs are given accordingly.

```
cd utils/text-ret/
python performRetrieval.py <path to data> <path to embedding>
  (or)
python performRetrieval.py <path to data> <path to P embedding> <path to R embedding> <path to S embedding>

For example:
python performRetrieval.py ./ cs_refined.bin
```

For any further information/questions, feel free to email the authors at *skottur@andrew.cmu.edu* or *vrama91@vt.edu*
