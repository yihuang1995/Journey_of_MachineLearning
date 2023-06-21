import numpy as np
from statistics import mode 
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from dtree import *
from sklearn.utils import resample
from collections import Counter


class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.oob_score = oob_score
        self.oob_score_ = np.nan
        self.trees = None

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data.  Keep track of the indexes of
        the OOB records for each tree.  After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        self.forest = []
        self.oob_index = []
        for i in range(self.n_estimators):
            tree = self.trees(self.min_samples_leaf,self.max_features)
            bag_idx = resample(range(len(X)))
            oob_idx = [x for x in range(len(X)) if x not in set(bag_idx)]
            bag_X = X[bag_idx]
            bag_y = y[bag_idx]
            tree.fit(bag_X,bag_y)
            self.forest.append(tree)
            self.oob_index.append(oob_idx)


        if self.oob_score:
            if self.trees == RegressionTree621: 
                oob_counts = np.zeros(len(X))
                oob_preds = np.zeros(len(X))             
                for i,tree_ in enumerate(self.forest):
                    tree_oob_index = self.oob_index[i]
                    for index in tree_oob_index:
                        oob_leaf = tree_.root.leaf(X[index])
                        nobs = oob_leaf.n
                        y_sum = nobs * oob_leaf.prediction
                        oob_counts[index] += nobs
                        oob_preds[index] += y_sum
                oob_avg_preds = oob_preds[oob_counts>0]/oob_counts[oob_counts>0]
                self.oob_score_ = r2_score(y[oob_counts > 0], oob_avg_preds)
            else:
                oob_counts = np.zeros(len(X))
                oob_preds = np.zeros((len(X),len(np.unique(y))))  
                for i,tree_ in enumerate(self.forest):
                    tree_oob_index = self.oob_index[i]
                    for index in tree_oob_index:
                        oob_leaf = tree_.root.leaf(X[index])
                        nobs = oob_leaf.n
                        tpred = oob_leaf.prediction
                        oob_preds[index,tpred] += nobs
                        oob_counts[index] += 1
                final_oob_index = np.where(oob_counts > 0)[0]
                oob_votes = oob_preds[final_oob_index].argmax(axis=-1)
                self.oob_score_ = accuracy_score(y[final_oob_index], oob_votes)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = RegressionTree621

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction.  Return a 1D vector
        with the predictions for each input record of X_test.
        """
        result = []
        for row in X_test:
            leaves = [tree_.root.leaf(row) for tree_ in self.forest]
            nobs = sum([leaf.n for leaf in leaves])
            y_sum = sum([leaf.n *leaf.prediction for leaf in leaves]) #np.sum(leaf.y)
            result.append(y_sum/nobs) 
        return np.array(result)
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the prediction for each record and then compute R^2 on that and y_test.
        """
        return r2_score(y_test, self.predict(X_test))
class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False):
        super().__init__(n_estimators, oob_score=oob_score)
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.trees = ClassifierTree621

    def predict(self, X_test) -> np.ndarray:
        result = []
        for row in X_test:
            counter = Counter()
            for sub_tree in self.forest:
                counter += Counter(sub_tree.root.leaf(row).y)
            result.append(counter.most_common(1)[0][0])
        return np.array(result)        
        
    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        return accuracy_score(y_test, self.predict(X_test)) 