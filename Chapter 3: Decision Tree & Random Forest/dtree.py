import numpy as np
from statistics import mode 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild

    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        else: return self.rchild.predict(x_test)
    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """
        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)
        else: 
            return self.rchild.leaf(x_test)

class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction
        self.y = y

    def predict(self, x_test):
        # return prediction
        return self.prediction

    def leaf(self, x_test):
        return self


class DecisionTree621:
    def __init__(self, min_samples_leaf=1,max_features = 0.3, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss # loss function; either np.std or gini
        self.max_features = max_features
    def fit(self, X, y):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.  
              
        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y)

    def bestsplit(self,X,y):
        best_col = best_split = -1
        best_loss = self.loss(y)
        num_feature = int(self.max_features * X.shape[1])
        vars = np.random.choice(range(X.shape[1]), num_feature ,replace= False)
        for column in vars:
            if len(X) > 25:
                candidates = np.random.choice(X[:,column], 15,replace= False)
            else:
                candidates = X[:,column]
            for split_val in candidates:
                yl = y[X[:,column] <= split_val]
                yr = y[X[:,column] > split_val]
                if len(yl) < self.min_samples_leaf or len(yr) < self.min_samples_leaf:
                    continue
                l = (len(yl) * self.loss(yl) + len(yr) * self.loss(yr)) / len(y)
                if l == 0: 
                    return column, split_val
                if l < best_loss:
                    best_col = column
                    best_split = split_val
                    best_loss = l
        return best_col, best_split


    def fit_(self, X, y):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.
        
        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) <= self.min_samples_leaf:
            return self.create_leaf(y)
        col,split = self.bestsplit(X,y)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:,col] <= split],y[X[:,col] <= split])
        rchild = self.fit_(X[X[:,col] > split],y[X[:,col] > split])
        return DecisionNode(col,split,lchild,rchild)
        
    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        result = []
        for item in X_test:
            result.append(self.root.predict(item))
        return result

    def create_leaf(self, y):
        return None

class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1,max_features = 0.3):
        super().__init__(min_samples_leaf=min_samples_leaf,max_features=max_features, loss=np.std)
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))
    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, prediction=np.mean(y))

def gini(y):
    "Return the gini impurity score for values in y"
    return 1 - sum((np.unique(y, return_counts=True)[1]/len(y))**2)

class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1,max_features = 0.3):
        super().__init__(min_samples_leaf=min_samples_leaf,max_features=max_features, loss=gini)
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))
    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, prediction=mode(y))