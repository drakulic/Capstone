from collections import Counter
import numpy as np

class TreeNode(object):
    '''
    A node class for a decision tree. It is either a leaf node or a parent node of two other nodes.

    This class has essentially one process, to predict a label for a row of data. There is also a
    method, as_string(), which is used to format a textual representation of the decision tree.
    Their dependency are as follows:

    Notation: f()<-g() means that the function f calls the function g.

    predict_one()|<-left.predict_one()
                 |<-right.predict_one()
                 |<-name
        If the node is a leaf, predict_one() returns `name` as the predicted label. Otherwise,
        the value of the feature is used to decide whether to go down the left or right branch
        of the node. Depending on the branch that is chosen, predict_one() returns the value
        from left.predict_one() or right.predict_one().

    as_string()|<-left.as_string()
               |<-right.as_string()
        as_string() formats a textual representation of the decision tree by recursively calling
        the representation of the left and right branches if the current node is not a leaf.
    '''
    def __init__(self):
        self.column = None  # (int)    index of feature to split on
        self.value = None  # value of the feature to split on
        self.categorical = True  # (bool) whether or not node is split on
                                 # categorial feature
        self.name = None    # (string) name of feature (or name of class in the
                            #          case of a list)
        self.left = None    # (TreeNode) left child
        self.right = None   # (TreeNode) right child
        self.leaf = False   # (bool)   true if node is a leaf, false otherwise
        self.classes = Counter()  # (Counter) only necessary for leaf node:
                                  #           key is class name and value is
                                  #           count of the count of data points
                                  #           that terminate at this leaf

    def predict_one(self, x):
        '''
        INPUT:
            x: A 1-d numpy array (single row of data)
        OUTPUT:
            y: A predicted label
            Return the predicted label for a single data point.
        '''
        if self.leaf:
            return self.name
        col_value = x[self.column]

        if self.categorical:
            if col_value == self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)
        else:
            if col_value < self.value:
                return self.left.predict_one(x)
            else:
                return self.right.predict_one(x)

    # This is for visualizing your tree. You don't need to look into this code.
    def as_string(self, level=0, prefix=""):
        '''
        INPUT:
            level: int (amount to indent)
        OUTPUT:
            prefix: str (to start the line with)
            Return a string representation of the tree rooted at this node.
        '''
        result = ""
        if prefix:
            indent = "  |   " * (level - 1) + "  |-> "
            result += indent + prefix + "\n"
        indent = "  |   " * level
        result += indent + "  " + str(self.name) + "\n"
        if not self.leaf:
            if self.categorical:
                left_key = str(self.value)
                right_key = "no " + str(self.value)
            else:
                left_key = "< " + str(self.value)
                right_key = ">= " + str(self.value)
            result += self.left.as_string(level + 1, left_key + ":")
            result += self.right.as_string(level + 1, right_key + ":")
        return result

    def __repr__(self):
        return self.as_string().strip()
