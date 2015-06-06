Tuple Extraction wrapper over Reverb 

Dependencies:
	1. Tested on Python 2.9, Requires docopt and textwrap installed via pip
	2. Requires Reverb code (Reverb-1.0\ 2/)
	
Usage:
	myTupleExtractor.py <dataset_name>

	Note, that there needs to be a file called "simplesentences_{$dataset_name}.json" in data folder for the code to run.

Output:
	Creates a variable in python environment called all_rels and all_nouns which lists all relations and all nouns, 
	along witht heir counts.

	count_rel and count_nouns is a dictionary with the key as the relation/noun and value as the count of how many times
	it occurs.
