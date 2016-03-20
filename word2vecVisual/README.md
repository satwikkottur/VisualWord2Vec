## How to use the code

The provided code trains vis-w2v. For the paper, checkout this 
[link](http://arxiv.org/abs/1511.07067). The code is organized as
follows:

* visword2vec.c: Main code for training vis-w2v
* refineFunctions.c : Contains functions related to refining embeddings
* helperFunctions.c : Mostly contain assisting functions like io, tokenization, etc.
* visualFunction.c : Contains code to refine based on tuples and also perform 
                        common sense (cs) task on test and validation sets
* vpFunctions.c : Contains code to refine based on sentences and also perform
                    visual paraphrasing (vp) task
* structs.h : Contains structures defined for the code
* filepaths.h: Contains the paths to the files needed for the above two tasks-cs,vp
* Makefile: Helps to setup, compile and run programs

Other files can be ignored for now.  

Program accepts following as inline arguments:
1. embed-path: Initialization for the embeddings (pre-trained using word2vec usually)  
    Format: Header should have <vocabsize> <dimensions>  
    Each following should first have the word, and feature vectors delimited by space  

1. output : Path to where to store the output embeddings
1. size : Size of the hidden layer (should match with the pre-loaded embeddings)
1. threads: Number of threads to use for refining, loading and other operations

Currently only saving one embedding is supported. For multi embeddings simply turn
the trainMulti flag (top of visword2vec file) to 1. However, saving can be done by
uncommenting code in trainModel().

### Changes to be made for usage:
1. Makefile
    * Liblinear-2.1 must be compiled and the path must be correctly set
    * yael must be setup (for k means) and corresponding paths setup
    Link here: [yael](https://gforge.inria.fr/projects/yael/)
    * cs and vp options should have correctly -embed-path options

1. filepaths.h:
    * Make sure all the paths are accessible and correctly set
    * Any change to this file, should be followed by re-compiling the code

To run either cs or vp, comment or uncomment corresponding wrapper calls in 
trainModel() function of visword2vec. And then `make cs` or `make vp` for the
two tasks correspondingly to compile and run. `make` simply compiles while 
`make clean` cleans up all the binaries.  

**NOTE**: All the binaries are stored in bin/ folder (might have to create one if 
doesnt exist beforehand).  

Contact : Satwik Kottur (skottur@andrew.cmu.edu) for further queries.
