from DecisionTree import DecisionTree
import numpy as np
from collections import Counter
from sklearn.cross_validation import train_test_split
import pandas as pd


class RandomForest(object):
    '''
    A Random Forest class.

    There are essentially three different processes: fitting, predicting and scoring.
    Their dependencies are as follows:

    Notation: f()<-g() means that the function f calls the function g.

    A. fit()<-build_forest()<-DecisionTree.DecisionTree()
        The fit method calls the build_forest method. The build_forest method creates
        a list of decision trees. Each decision tree is trained on bootstrapped data
        and on a random subset of features.

    B. predict()
        This essentially calls each decision tree in a list of decision trees and returns
        the most common answer.

    C. score()<-predict()
        This calculates the accuracy of the model on a given dataset.
    '''

    def __init__(self, num_trees, num_features):
        '''
        Input:
            num_trees:  The number of trees to create in the forest.
            num_features:  The number of features to consider when choosing the
                           best split for each node of the decision trees.
        '''
        self.num_trees = num_trees
        self.num_features = num_features
        self.forest = None

    def fit(self, X, y):
        '''
        Input:
            X: A 2-d numpy array. Eeach column is feature and each row a data point.
            y: A 1-d numpy array. Each value is a label for a data point.
        '''
        self.forest = self.build_forest(X, y, self.num_trees, X.shape[0], \
                                        self.num_features)

    def build_forest(self, X, y, num_trees, num_samples, num_features):
        '''
        Input:
            X:  A 2-d numpy array. Eeach column is feature and each row a data point.
            y:  A 1-d numpy array. Each value is a label for a data point.
            num_trees:  The number of trees to create in the forest.
            num_samples:  The size of the re-sampled data sets.
            num_features:  The number of features to consider when choosing the
                           best split for each node of the decision trees.

        Output:
            Returns a list of num_trees DecisionTrees.
        '''
        forest = []
        for i in xrange(num_trees):
            sample_indices = np.random.choice(X.shape[0], num_samples, \
                                              replace=True)
            sample_X = np.array(X[sample_indices])
            sample_y = np.array(y[sample_indices])
            dt = DecisionTree(num_features=self.num_features)
            dt.fit(sample_X, sample_y)
            forest.append(dt)
        return forest

    def predict(self, X):
        '''
        Input:
            X:  A 2-d numpy array. Eeach column is feature and each row a data point.

        Output:
            Return a numpy array of the labels predicted for the given test data.
        '''
        answers = np.array([tree.predict(X) for tree in self.forest]).T
        return np.array([Counter(row).most_common(1)[0][0] for row in answers])

    def score(self, X, y):
        '''
        Input:
            X:  A 2-d numpy array. Eeach column is feature and each row a data point.
            y:  A 1-d numpy array. Each value is a label for a data point.

        Output:
            Returns the accuracy of the Random Forest for the given test data and
        labels.
        '''
        return sum(self.predict(X) == y) / float(len(y))

if __name__ == '__main__':
    pass
