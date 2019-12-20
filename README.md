# DAT620_delivered
Classifying nordic language news articles into hierarchical topics

## Overview of files:

### preprocessing.ipynb

This file contain the code for making the Pandas DataFrame that is used throughout the project. It needs access to the data_hu.csv file for labels, access to the /data_hu/ directory with the articles and access to the fasttext model 81.

### nb_nn_classifier.ipynb

This file contain the code for identifying nynorsk articles and appending a boolean column indicating nynorsk to the Pandas DataFrame.


### analysis.ipynb

This file contain the code for doing TFIDF word analysis on the corpus and classes, aswell as code for doing class by class feature selection for creating custom vocabulary.

### pos_tagging.ipynb

This file contain code for doing POS tagging, and ontology tagging. It also contain for creating a synonym dictionary not implemented in the project. It also does aggregation of the POS tags and ontology tags and appends columns to the Pandas DataFrame.

It requires access to the THE NORWEGIAN WORDNET BOKMÅL  v.1.1.2 files.

### modeling.ipynb

This file contain code for doing gridsearch of hyperparameters and comparing classifiers, samplers, etc. It is the workhorse of the project. It requires access to the modeling_tools.py.

### modeling_tools.py

This file contain functions for iterating over all model constelations that is specified in the modeling.ipynb file that calls it. It also contain code for plotting a confusion matrix.

### bucketing.ipynb

This file contain code for creating the BucketClassifier and running a classification example on the corpus.

### ensemble.ipynb

This file contain code for creating the EnsembleClassifier and running a classification example on the corpus.

## Additional files

### presentation.ipynb

A early presentation with some statistics and plots about the corpus and different constelations of models performance.

### presentation_tools.py

Methods for applying plots and statistics in the presentation.ipynb file.

### BERT.ipynb

This file contain code for finetuning a BERT model using an sklearn wrapper created by bert_sklearn. It failed to run locally due to memory error. But could be interesting to run on a better machine.