import pandas as pd
import numpy as np
import math
from collections import Counter
from TreeNode import TreeNode


class DecisionTree(object):
    '''
    A decision tree class.

    There are essentially two different processes: fitting and predicting.
    Their dependencies are as follows:

    Notation: f()<-g() means that the function f calls the function g.

    fit()<-_build_tree()|<-_build_tree()
                        |<-TreeNode.TreeNode()
                        |<-choose_split_index()|<-_make_split()
                                               |<-_information_gain()<-impurity_criteria()

        The fit method processes the labeled data, assigns labels and figures out if it is
        categorical. It then passes the data to the _build_tree method.

        The _build_tree method chooses the best feature to split on by calling
        choose_split_index(). It then creates a new tree node based on this split. If their
        are no further features to split on then the node is a leaf. Otherwise we call
        _build_tree on on each branch using the subset of features and data for each branch.

        The choose_split_index method iterates through each feature that has not yet been
        split on and calculates the information_gain(). It then returns the best feature
        with the highest information gain.

        The information_gain method calls the impurity_criteria method which is initialized
        in the constructor to be the _entropy method or the _gini method.

    predict()<-TreeNode.predict_one
        The predict method calls the predict_one method on the root node for each row of
        data.
    '''

    def __init__(self, impurity_criterion='entropy', num_features=None):
        '''
        Initialize an empty DecisionTree.
        '''

        self.root = None  # root Node
        self.feature_names = None  # string names of features (for interpreting
                                   # the tree)
        self.categorical = None  # Boolean array of whether variable is
                                 # categorical (or continuous)
        self.impurity_criterion = self._entropy \
                                  if impurity_criterion == 'entropy' \
                                  else self._gini
        self.num_features = num_features

    def fit(self, X, y, feature_names=None):
        '''
        INPUT:
            X: 2-d numpy array. Each column is a feature and each row a data point.
            y: 1-d numpy array. Each value is a label for a data point.
            feature_names: numpy array of strings. It is an optional list contianing the names of each of the features.
        OUTPUT: None
            Build the decision tree.
        '''

        if feature_names is None or len(feature_names) != X.shape[1]:
            self.feature_names = np.arange(X.shape[1])
        else:
            self.feature_names = feature_names

        # Create True/False array of whether the variable is categorical
        #is_categorical = lambda x: isinstance(x, str) or \
        #                           isinstance(x, bool) or \
        #                           isinstance(x, unicode)
        #self.categorical = np.vectorize(is_categorical)(X[0])

        self.root = self._build_tree(X, y)
        if not self.num_features:
            self.num_features = X.shape[1]

    def _build_tree(self, X, y):
        '''
        INPUT:
            X: A 2-d numpy array. Eeach column is feature and each row a data point.
            y: A 1-d numpy array. Each value is a label for a data point.
        OUTPUT:
            TreeNode
            Recursively build the decision tree. Return the root node.
        '''

        node = TreeNode()
        index, value, splits = self._choose_split_index(X, y)

        if index is None or len(np.unique(y)) == 1:
            node.leaf = True
            node.classes = Counter(y)
            node.name = node.classes.most_common(1)[0][0]
        else:
            X1, y1, X2, y2 = splits
            node.column = index
            node.name = self.feature_names[index]
            node.value = value
            node.categorical = self.categorical[index]
            node.left = self._build_tree(X1, y1)
            node.right = self._build_tree(X2, y2)
        return node

    def _entropy(self, y):
        '''
        INPUT:
            y:  A 1-d numpy array. Each value is a label for a data point.
        OUTPUT:
            The entropy of y.
        '''

        total = 0
        for cl in np.unique(y):
            prob = np.sum(y == cl) / float(len(y))
            total += prob * math.log(prob)
        return -total

    def _gini(self, y):
        '''
        INPUT:
            y:  A 1-d numpy array. Each value is a label for a data point.
        OUTPUT:
            The gini impurity of y.
        '''

        total = 0
        for cl in np.unique(y):
            prob = np.sum(y == cl) / float(len(y))
            total += prob ** 2
        return 1 - total

    def _make_split(self, X, y, split_index, split_value):
        '''
        INPUT:
            X: A 2-d numpy array. Eeach column is feature and each row a data point.
            y: A 1-d numpy array. Each value is a label for a data point.
            split_index: int (index of feature)
            split_value: int/float/bool/str (value of feature)
        OUTPUT:
            X1: 2-d numpy array (feature matrix for subset 1).
            y1: 1-d numpy array (labels for subset 1).
            X2: 2-d numpy array (feature matrix for subset 2).
            y2: 1-d numpy array (labels for subset 2).

            Return the two subsets of the dataset achieved by the given feature and
            value to split on.

            Call the method like this:
            >>> X1, y1, X2, y2 = self._make_split(X, y, split_index, split_value)

            X1, y1 is a subset of the data.
            X2, y2 is the other subset of the data.
        '''

        split_col = X[:, split_index]
        if isinstance(split_value, int) or isinstance(split_value, float):
            A = split_col < split_value
            B = split_col >= split_value
        else:
            A = split_col == split_value
            B = split_col != split_value
        return X[A], y[A], X[B], y[B]

    def _information_gain(self, y, y1, y2):
        '''
        INPUT:
            y: A 1-d numpy array. Each value is a label for a data point.
            y1: A 1-d numpy array. Each value is a label for a data point in subset 1.
            y2: A 1-d numpy array. Each value is a label for a data point in subset 2.
        OUTPUT:
            Return the information gain of making the given split.
            Use self.impurity_criterion(y) rather than calling _entropy or _gini
            directly.
        '''

        total = self.impurity_criterion(y)
        for y_split in (y1, y2):
            ent = self.impurity_criterion(y_split)
            total -= len(y_split) * ent / float(len(y))
        return total

    def _choose_split_index(self, X, y):
        '''
        INPUT:
            X: A 2-d numpy array. Eeach column is feature and each row a data point.
            y: A 1-d numpy array. Each value is a label for a data point.
        OUTPUT:
            index: int (index of feature)
            value: int/float/bool/str (value of feature)
            splits: (2d array, 1d array, 2d array, 1d array)

            Determine which feature and value to split on. Return the index and
            value of the optimal split along with the split of the dataset.

            Return None, None, None if there is no split which improves information
            gain.

            Call the method like this:
            >>> index, value, splits = self._choose_split_index(X, y)
            >>> X1, y1, X2, y2 = splits
        '''

        split_index, split_value, splits = None, None, None
        feature_indices = np.random.choice(X.shape[1], self.num_features, replace=False)
        max_gain = 0
        X = X[:, feature_indices]

        for i in xrange(X.shape[1]):
            values = np.unique(X[:, i])
            if len(values) < 2:
                continue
            for val in values:
                temp_splits = self._make_split(X, y, i, val)
                X1, y1, X2, y2 = temp_splits
                gain = self._information_gain(y, y1, y2)
                if gain > max_gain:
                    max_gain = gain
                    split_index, split_value = i, val
                    splits = temp_splits
        return split_index, split_value, splits

    def predict(self, X):
        '''
        INPUT:
            X: A 2-d numpy array. Eeach column is feature and each row a data point.
        OUTPUT:
            y: A 1-d numpy array. Each value is a label for a data point.
            Return an array of predictions for the feature matrix X.
        '''
        return np.apply_along_axis(self.root.predict_one, axis=1, arr=X)

    def __str__(self):
        '''
        Return string representation of the Decision Tree.
        '''
        return str(self.root)
