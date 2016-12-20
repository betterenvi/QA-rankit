class ParamGA(object):
    def __init__(self, QID_trn, X_trn, Y_trn, QID_dev, X_dev, Y_dev, 
                 model_name='RandomForestClassifier', param_funcs=dict(), 
                 param_static=dict(), param_index=dict(),
                 pair_wise=False):
        self.QID_trn, self.X_trn, self.Y_trn = QID_trn, X_trn, Y_trn
        self.QID_dev, self.X_dev, self.Y_dev = QID_dev, X_dev, Y_dev
        self.rank_dev = DataFrame({}, index=self.X_dev.index)
        self.rank_dev['Label'] = self.Y_dev
        self.rank_dev['QID'] = self.QID_dev
        self.counter = itertools.count()
        self.mrrs = dict()
        self.model_name = model_name
        self.param_funcs = param_funcs
        self.param_static = param_static
        self.param_index = param_index
        self.pair_wise = pair_wise
        if self.pair_wise:
            self.pair_ranker = PairWiseRanker(self.QID_trn, self.X_trn, self.Y_trn)
            self.pair_ranker.init_predict(self.QID_dev, self.X_dev)
        else:
            self.models = dict()
        
    def _gen_param(self):
        param = [None] * len(self.param_index)
        for pn, func in self.param_funcs.items():
            param[self.param_index[pn]] = func['gen']()
        return param

    def _evaluate(self, indiv):
        eval_str = self.model_name + '('
        flag = ''
        for pn in self.param_funcs:
            gene = indiv[self.param_index[pn]]
            if type(gene) in [str, np.string_]:
                eval_str += flag + pn + '="' + str(gene) + '"'
            else:
                eval_str += flag + pn + '=' + str(gene)
            flag = ', '
        for pn in self.param_static:
            if type(self.param_static[pn]) in [str, np.string_]:
                eval_str += flag + pn + '="' + str(self.param_static[pn]) + '"'
            else:
                eval_str += flag + pn + '=' + str(self.param_static[pn])
            flag = ', '
        eval_str += ')'
        #print eval_str
        model = eval(eval_str)
        model_idx = self.counter.next()
        if self.pair_wise:
            self.pair_ranker.fit(model, model_idx)
            pred = self.pair_ranker.do_predict(model_idx)
        else:
            model.fit(self.X_trn, self.Y_trn)
            self.models[model_idx] = model
            pred = Series(model.predict_proba(self.X_dev)[:, 1], index=self.X_dev.index)
        self.rank_dev['pred'] = pred # pred must be a Series, not an array
        self.rank_dev.sort_values(['QID', 'pred'], inplace=True, ascending=False)
        grp = self.rank_dev.Label.groupby(self.rank_dev.QID)
        mrr = MRR(grp, keep_no_ans=False)
        self.mrrs[model_idx] = mrr
        print '    >', model_idx, np.round(mrr, 4), indiv
        return mrr, 

    def _mut_indiv(self, indiv, indiv_pb):
        for pn, func in self.param_funcs.items():
            if random.random() < indiv_pb:
                indiv[self.param_index[pn]] = func['mut']()

    def run(self, NPOP=30, NGEN=10, CXPB=0.5, MUTPB=0.2):
        self.ga = MyGA(self._gen_param, self._evaluate, self._mut_indiv, CXPB=CXPB, MUTPB=MUTPB)
        self.ga.init_pop(NPOP=NPOP)
        self.ga.iterate(NGEN=NGEN)
        

class RFGA(ParamGA):
    ''' RandomForestClassifier
    n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None
    '''
    param_names = ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    param_funcs = collections.defaultdict(dict)
    param_funcs['n_estimators']['gen'] = lambda : random.randint(20, 2001)
    param_funcs['n_estimators']['mut'] = lambda : random.randint(20, 2001)
    param_funcs['criterion']['gen'] = lambda : random.choice(['gini', 'entropy'])
    param_funcs['criterion']['mut'] = lambda : random.choice(['gini', 'entropy'])
    param_funcs['max_depth']['gen'] = lambda : random.choice([None] + range(3, 21))
    param_funcs['max_depth']['mut'] = lambda : random.choice([None] + range(3, 21))
    param_funcs['min_samples_split']['gen'] = lambda : random.randint(2, 6)
    param_funcs['min_samples_split']['mut'] = lambda : random.randint(2, 6)
    param_funcs['min_samples_leaf']['gen'] = lambda : random.randint(1, 6)
    param_funcs['min_samples_leaf']['mut'] = lambda : random.randint(1, 6)
    def __init__(self, *args, **kwargs):
        kwargs['model_name'] = 'RandomForestClassifier'
        kwargs['param_funcs'] = RFGA.param_funcs
        kwargs['param_static'] = {'n_jobs':-1}
        kwargs['param_index'] = {k:i for i, k in enumerate(RFGA.param_names)}
        super(RFGA, self).__init__(*args, **kwargs)
    