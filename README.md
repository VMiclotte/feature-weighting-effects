# ML-project

## Usage:
Adjust the variables in main function in Main.java as follows.

path: path to folder with folds
When making folds the parent directory has to contain the .arff file with all data
e.g. folder Data contains heart.arff and a subfolder heart which the folds will be saved into

numFolds: number of folds

k: number of nearest neighbours considered in the algorithms

useNN: If this is set to true, k nearest neighbours will be considered in the algorithms.
Otherwise, all training instances will be used as "nearest neighbours" in the algorithms.

discretize: If this is set to true, the data will be discretized using k-means.
Otherwise the raw data is used.

makeFolds: If this is set to true, folds will be created from the file (path + ".arff").
Otherwise folds are assumed to exist in the directory with path as given path.
Note: when using discretization, different files with different filenames will be created.