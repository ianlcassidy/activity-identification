# Activity Identification

This repository contains the code used to investigate supervised learning algorithms for activity identification from sensor data.  The data was downloaded from the UC Irvine Machine Learning Repository:
[http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring](http://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

The `data_processing.py` should be run first.  This script outputs the training data in the form of CSV's.  In order to run this script, the PAMAP2_Dataset folder should be placed in the same directory as the `data_processing.py` script (or the script should be edited with the path to this folder).

The `activity_identification_ML.ipynb` file is an iPython Notebook file containing the machine learning analysis.  It requires the `X_train.csv` and `Y_train.csv` files as inputs.
