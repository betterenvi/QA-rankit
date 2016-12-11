import sys, os, collections, copy
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

class PairWiseRanker(object):
    """docstring for PairWiseRanker"""
    def __init__(self, QID, X, Y):
        super(PairWiseRanker, self).__init__()
        self.QID, self.X, self.Y = QID, X, Y
        self.N, self.M = X.shape
        self.X_new, self.Y_new = self._reconstruct_features(QID, X, Y)
        self.models = dict()

    def _reconstruct_features(self, QID, X, Y=np.array([])):
        has_Y = (len(Y) > 0)
        if not has_Y:
            Y = np.zeros(X.shape[0])
        Y_opp = np.ones(len(Y)) - Y
        X_tmp = copy.deepcopy(X)
        X_tmp['QID'] = QID
        X_tmp['QAID'] = X_tmp.index
        X_tmp['tmp_idx'] = np.arange(0, len(Y))
        X_tmp['Y'] = Y
        X_tmp['Y_opp'] = Y_opp # construct new row from two old row with different y
        if has_Y:
            X_new = pd.merge(X_tmp, X_tmp, how='inner', left_on=['QID', 'Y'], right_on=['QID', 'Y_opp'], suffixes=('_l', '_r'))
        else:
            X_new = pd.merge(X_tmp, X_tmp, how='inner', left_on=['QID'], right_on=['QID'], suffixes=('_l', '_r'))
        #X_new = X_new[X_new['tmp_idx_l'] < X_new['tmp_idx_r']]
        X_new.index = X_new['QAID_l'] + ' ' + X_new['QAID_r']
        Y_new = X_new['Y_l']
        cols_to_drop = ['QAID', 'Y', 'Y_opp', 'tmp_idx']
        lcols_to_drop = map(lambda col : col + '_l', cols_to_drop)
        rcols_to_drop = map(lambda col : col + '_r', cols_to_drop)
        X_new.drop(['QID'] + lcols_to_drop + rcols_to_drop, axis=1, inplace=True)
        Y_new = Y_new.loc[X_new.index]
        X_new = X_new
        Y_new = Y_new
        return X_new, Y_new

    def fit(self, model, mn):
        model.fit(self.X_new, self.Y_new)
        self.models[mn] = model
        self.model = model
        return self

    def predict(self, QID, X, mn):
        self.model = self.models[mn]
        X_new, _ = self._reconstruct_features(QID, X)
        pred_new = self.model.predict_proba(X_new)
        pred = Series([0.0] * X.shape[0], index=X.index)
        qaid_pairs = map(lambda pair : pair.split(' '), X_new.index)
        qaid_l = map(lambda pair : pair[0], qaid_pairs)
        qaid_r = map(lambda pair : pair[1], qaid_pairs)
        pred_l = Series(pred_new[:, 1], index=qaid_l)
        pred_r = Series(pred_new[:, 0], index=qaid_r)
        pred += pred_l.groupby(pred_l.index).agg(sum)
        pred += pred_r.groupby(pred_r.index).agg(sum)
        # pred[qaid_l] += pred_new[:, 1] # wrong. seems qaid_l may has two or more elements that have same values
        # pred[qaid_r] += pred_new[:, 0]
        return pred.values
