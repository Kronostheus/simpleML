# simpleML
Very crude implementations of common machine learning algorithms.


These are some basic implementations of machine learning algorithms from scratch, mostly using numpy/scipy. They are designed as a learning experiment and are not suitable for general use other than that.

Currently, this repository contains the following directories:

* `superpvised/`
  * `knn.py` - K Nearest Neighbours Classifier
  * `simple_linear_regression.py` - Linear Regression designed for single dependent and indepent variable
  * `gaussian_naive_bayes.py` - Gaussian Naive Bayes classifier
  * `perceptron.py` - Single Layer Perceptron
  * `logistic_regression.py` - Logistic Regression classifier
  * `decision_tree_id3.py` - Decision Tree using ID3 algorithm
* `unsupervised/`
  * `kmeans.py` - simple KMeans
  * `dbscan.py` - DBSCAN
  * `mean_shift.py` - Mean Shift
  * `spectral_clustering.py` - Spectral Clustering
* `utils/`
  * `metrics.py` - system evaluation metrics
  * `calculations.py` - distances and other math related calculations
