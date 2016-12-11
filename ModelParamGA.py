import sys, os, collections, copy, re, random, itertools
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from util import *
from PairWiseRanker import *
from sklearn.ensemble import RandomForestClassifier
from MyGA import *


class RFGA(object):
    ''' RandomForestClassifier
    n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None
    '''
    gen_n_tree = lambda : random.randint(20, 2001)
    gen_criterion = lambda : random.choice(['gini', 'entropy'])
    gen_max_depth = lambda : random.choice([None] + range(3, 21))
    gen_min_samples_split = lambda : random.randint(2, 6)
    gen_min_samples_leaf = lambda : random.randint(1, 6)
    gen_funcs = [gen_n_tree, gen_criterion, gen_max_depth, gen_min_samples_split, gen_min_samples_leaf]

    def __init__(self, QID_trn, X_trn, Y_trn, QID_dev, X_dev, Y_dev):
        self.QID_trn, self.X_trn, self.Y_trn = QID_trn, X_trn, Y_trn
        self.QID_dev, self.X_dev, self.Y_dev = QID_dev, X_dev, Y_dev
        self.rank_dev = DataFrame({}, index=self.X_dev.index)
        self.rank_dev['Label'] = self.Y_dev
        self.rank_dev['QID'] = self.QID_dev
        self.pair_ranker = PairWiseRanker(self.QID_trn, self.X_trn, self.Y_trn)
        self.pair_ranker.init_predict(self.QID_dev, self.X_dev)
        self.counter = itertools.count()
        self.mrrs = dict()
        
    def _gen_param(self):
        param = list()
        for func in RFGA.gen_funcs:
            param.append(func())
        return param

    def _evaluate(self, indiv):
        rf = RandomForestClassifier(n_estimators=indiv[0], criterion=indiv[1], 
                                    max_depth=indiv[2], min_samples_split=indiv[3], 
                                    min_samples_leaf=indiv[4], n_jobs=-1)
        model_name = self.counter.next()
        self.pair_ranker.fit(rf, model_name)
        pred = self.pair_ranker.do_predict(model_name)
        self.rank_dev['pred'] = pred # pred must be a Series, not an array
        self.rank_dev.sort_values(['QID', 'pred'], inplace=True, ascending=False)
        grp = self.rank_dev.Label.groupby(self.rank_dev.QID)
        mrr = MRR(grp, keep_no_ans=False)
        self.mrrs[model_name] = mrr
        print '    >', model_name, np.round(mrr, 4), indiv
        return mrr, 

    def _mut_indiv(self, indiv, indiv_pb):
        for i, ele in enumerate(indiv):
            if random.random() < indiv_pb:
                print '    mut'
                indiv[i] = RFGA.gen_funcs[i]()
                # tp = type(indiv[i])
                # if i == 0:
                #     indiv[i] = gen_funcs[i]()
                # elif i == 1:
                #     indiv[i] = max(1, indiv[i] + random.choice([1, -1]))
                # elif i == 2:
                #     indiv[i] = min(0.9, indiv[i] * (random.random() + 0.5))
                # elif i == 3:
                #     indiv[i] = max(0, indiv[i] + random.choice([1, -1]))

    def run(self, NPOP=30, NGEN=10, CXPB=0.5, MUTPB=0.2):
        self.ga = MyGA(self._gen_param, self._evaluate, self._mut_indiv, CXPB=CXPB, MUTPB=MUTPB)
        self.ga.init_pop(NPOP=NPOP)
        self.ga.iterate(NGEN=NGEN)