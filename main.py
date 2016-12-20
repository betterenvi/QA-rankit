# coding: utf-8
# Please use Python 2.
# Check if all the required packages are installed.
# Please use Theano as backend of Keras.
# Using IPython or Jupyter Notebook to run this code is strongly recommended!
# Run line by line or block by block, and you can config some important parameters
# and output whatever that you want!
import sys, os, collections, re, commands, csv
import nltk, codecs
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from copy import deepcopy
from util import *
from PairWiseRanker import *

UNKNOWN_WORD = ':-)'

# preparing

# config data path
data_dir = 'data/'
fn = {'trn':data_dir + 'WikiQA-train.tsv', 'dev':data_dir + 'WikiQA-dev.tsv', 'test':data_dir + 'WikiQA-test.tsv'}

# config word2vec
word2vec_dir = '/store1/chenqy/'
EMBEDDING_DIM = 50
word2vec_fn = word2vec_dir + 'glove.6B.%dd.txt' % EMBEDDING_DIM

# read data
def read_data(f):
    data = pd.read_csv(f, sep='\t', header=0, quoting=csv.QUOTE_NONE)
    data['QAID'] = data['QuestionID'] + '*' + data['SentenceID']
    data.index = data['QAID']
    for col in ['Question', 'Sentence']:
        data[col] = map(lambda x : codecs.decode(x.lower(), 'UTF-8'), data[col])
    questions = dict(zip(data['QuestionID'], data['Question']))
    return data, questions

data, questions = {}, {}
for k in fn:
    data[k], questions[k] = read_data(fn[k])
    print k, data[k].shape, len(set(data[k]['QAID']))

data_trn, data_dev, data_test = data['trn'], data['dev'], data['test'] # for convienience

# extract features

# some preparations: tokenize, stem, drop stopwords, lemma
wnl = WordNetLemmatizer()
regex_tokenizer = RegexpTokenizer(r'\w+')
lancaster = nltk.LancasterStemmer()
def sentence2words(sent):
    res = {}
    res['tokens'] = nltk.word_tokenize(sent)
    res['words'] = regex_tokenizer.tokenize(sent)
    res['lemma'] = [wnl.lemmatize(w) for w in res['words']]
    res['stems'] = [lancaster.stem(w) for w in res['words']]
    res['words_not_stop'] = filter(lambda word : word not in stopwords.words('english'), res['words'])
    res['stems_not_stop'] = [lancaster.stem(w) for w in res['words_not_stop']]
    res['lemma_not_stop'] = [wnl.lemmatize(w) for w in res['words_not_stop']]
    return res

tps = ['words', 'stems', 'words_not_stop', 'stems_not_stop', 'lemma', 'lemma_not_stop']

sent_words = dict()
for k in fn:
    for qid, q in questions[k].items():
        sent_words[qid] = sentence2words(q)
    d = data[k]
    for qaid in d.index:
        sent_words[qaid] = sentence2words(d.get_value(qaid, 'Sentence'))

first_token = lambda tokens : tokens[0]
second_token = lambda tokens : UNKNOWN_WORD if len(tokens) == 1 else tokens[1]
first_two_tokens = lambda tokens : ' '.join(tokens[:2])

for k in fn:
    for col, func in [('token1', first_token), ('token2', second_token), ('token12', first_two_tokens)]:
        qids = list(set(data[k]['QuestionID']))
        tmp = Series(map(func, map(lambda qid : sent_words[qid]['words'], qids)), index=qids)
        data[k][col] = tmp[data[k]['QuestionID']].values


def get_question_type(data_x):
    qtype = deepcopy(Series(data_x['token1'], index=data_x.index))
    qtype[data_x['token12'] == 'what year'] = 'time'
    qtype[data_x['token1'] == 'when'] = 'time'
    qtype[data_x['token12'] == 'how many'] = 'number'
    qtype[data_x['token1'] == 'where'] = 'place'
    qtype[data_x['token1'] == 'who'] = 'person'
    return qtype

for k in fn:
    data[k]['qtype'] = get_question_type(data[k])


import math

words_idf = dict()
for tp in tps:
    words_idf[tp] = dict()

for tp in tps:
    for qaid in sent_words:
        for word in set(sent_words[qaid][tp]):
            if (word in words_idf[tp]) == True:
                words_idf[tp][word] = words_idf[tp][word] + 1
            else:
                words_idf[tp][word] = 1

DocNum = len(sent_words)
for tp in tps:
    for w in words_idf[tp]:
        words_idf[tp][w] = math.log(DocNum / words_idf[tp][w], 2)

def count_num_cooccur_weight(x, y, tp):
    z = set(x) & set(y)
    sum = 0
    for w in z:
        sum = sum + words_idf[tp][w]
    return sum

# word embedding
# readEmbedFile
from word2vec_util import *
wordvecs = readEmbedFile(word2vec_fn)
dim = EMBEDDING_DIM
wordset2 = set()
for tp in tps:
    for w in words_idf[tp]:
        wordset2.add(w)

new_wv=dict()
for w in wordvecs:
    if (w in wordset2) == True:
        new_wv[w] = wordvecs[w]
wordvecs = new_wv
new_wv = dict()

def count_wv_dis(x, y):
    vx = [0 for c in range(dim)]
    wn = 0
    for w in x:
        if w in wordvecs:
            wn = wn + 1
            for d in range(dim):
                vx[d] += wordvecs[w][d]
    if wn > 0:
        for d in range(dim):
            vx[d] /= wn
    vy = [0 for c in range(dim)]
    wn = 0
    for w in y:
        if w in wordvecs:
            wn = wn + 1
            for d in range(dim):
                vy[d] += wordvecs[w][d]
    if wn > 0:
        for d in range(dim):
            vy[d] /= wn
    dis = 0;
    for d in range(dim):
        dis += (vx[d] - vy[d])*(vx[d] - vy[d])
    return 0 - dis # larger, better

def count_wv_dis_weight(x, y, tp):
    vx = [0 for c in range(dim)]
    wn = 0
    for w in x:
        if w in wordvecs:
            wn = wn + words_idf[tp][w]
            for d in range(dim):
                vx[d] += wordvecs[w][d]*words_idf[tp][w]
    if wn > 0:
        for d in range(dim):
            vx[d] /= wn
    vy = [0 for c in range(dim)]
    wn = 0
    for w in y:
        if w in wordvecs:
            wn = wn + words_idf[tp][w]
            for d in range(dim):
                vy[d] += wordvecs[w][d]*words_idf[tp][w]
    if wn > 0:
        for d in range(dim):
            vy[d] /= wn
    dis = 0;
    for d in range(dim):
        dis += (vx[d] - vy[d])*(vx[d] - vy[d])
    return 0 - dis # larger, better

# index word
# encode unicode str as utf-8. otherwise it will cause err
encode = lambda s : codecs.encode(s, 'UTF-8')
ques_utf8 = {}
sent_utf8 = {}
for k in fn:
    for qid, q in questions[k].items():
        ques_utf8[qid] = encode(q)
    d = data[k]
    for qaid in d.index:
        sent_utf8[qaid] = encode(d.get_value(qaid, 'Sentence'))
texts = []
texts.extend(ques_utf8.values())
texts.extend(sent_utf8.values())
tokenizer = Tokenizer(nb_words=None)
tokenizer.fit_on_texts(texts)
ques_seq = {qid:tokenizer.texts_to_sequences([tx])[0] for qid, tx in ques_utf8.items()}
sent_seq = {qaid:tokenizer.texts_to_sequences([tx])[0] for qaid, tx in sent_utf8.items()}
word_index = tokenizer.word_index

word2vec = pd.read_csv(word2vec_fn, sep=' ', header=None, index_col=0, nrows=None, quoting=csv.QUOTE_NONE)
embedding_index = {}
for i, w in enumerate(word2vec.index):
    embedding_index[w] = word2vec.values[i]

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

words_bag = set(word2vec.index)


words_idf_series = {tp:Series(words_idf[tp].values(), index=words_idf[tp].keys()) for tp in words_idf}

def words_of_sentence_to_vector(words, tp, use_weight=True):
    shp = (len(words), EMBEDDING_DIM)
    try:
        if use_weight:
            weight = np.repeat(words_idf_series[tp].loc[words], EMBEDDING_DIM).values.reshape(shp)
            return (word2vec.loc[words] * weight).mean().values
        else:
            return (word2vec.loc[words]).mean().values
    except: # none of words are in word2vec
        return np.zeros(shp)

def get_sent_vetor_df(q_words):
    q_vecs = dict(zip(q_words.keys(), map(lambda words : words_of_sentence_to_vector(words, 'words', use_weight=False), q_words.values())))
    q_vecs = DataFrame(q_vecs).T
    q_vecs_wgt = dict(zip(q_words.keys(), map(lambda words : words_of_sentence_to_vector(words, 'words', use_weight=True), q_words.values())))
    q_vecs_wgt = DataFrame(q_vecs_wgt).T
    #q_vecs_wgt.columns = map(lambda col : col)
    q_vecs_merged = pd.concat([q_vecs, q_vecs_wgt], axis=1)
    return q_vecs_merged


# calculate features

features = {key:DataFrame({}, index=data[key].index) for key in fn}
features_trn = features['trn']
features_dev = features['dev']
features_test = features['test']

tps_wgt = map(lambda v : v + '_wgt', tps)
tps_wv = map(lambda v : v + '_wv', tps)
tps_wv_wgt = map(lambda v : v + '_wgt', tps_wv)

# words, stems
for k in fn:
    d = data[k]
    for tp in tps:
        func = lambda qaid : count_num_cooccur(sent_words[qaid][tp], sent_words[d.get_value(qaid, 'QuestionID')][tp])
        features[k][tp] = d['QAID'].apply(func)
        func = lambda qaid : count_num_cooccur_weight(sent_words[qaid][tp], sent_words[d.get_value(qaid, 'QuestionID')][tp], tp)
        features[k][tp + '_wgt'] = d['QAID'].apply(func)
        func = lambda qaid : count_wv_dis(sent_words[qaid][tp], sent_words[d.get_value(qaid, 'QuestionID')][tp])
        features[k][tp + '_wv'] = d['QAID'].apply(func)
        func = lambda qaid : count_wv_dis_weight(sent_words[qaid][tp], sent_words[d.get_value(qaid, 'QuestionID')][tp], tp)
        features[k][tp + '_wv_wgt'] = d['QAID'].apply(func)

# question type
for k in fn:
    f, d = features[k], data[k]
    for col in ['token1', 'qtype']:
        f[col] = data[k][col]
    f['qtype_time'] = (d['qtype'] == 'time') + 0
    f['qtype_number'] = (d['qtype'] == 'number') + 0
    f['year'] = map(lambda sent : int(re.search('[0-9]{4}', sent) != None), d.Sentence)
    f['number'] = map(lambda sent : int(re.search('[0-9]+', sent) != None), d.Sentence)
    f['number'] = f['number'] - f['year']
    f['num_unknown'] = map(lambda words : len(set(words) - words_bag), #np.sum(np.logical_not(np.in1d(words, word2vec.index)))
                           map(lambda qaid : sent_words[qaid]['words'], d.index))

# encode questions and sentences as vector. give up finally
# qa_vectors = {}
# for k in fn:
#     q_words = {qid:sent_words[qid]['words'] for qid in set(data[k]['QuestionID'])}
#     q_vec_df = get_sent_vetor_df(q_words)
#     q_vec_df = q_vec_df.loc[data[k]['QuestionID']]
#     q_vec_df.index = data[k].index
#     a_words = {qaid:sent_words[qaid]['words'] for qaid in data[k].index}
#     a_vec_df = get_sent_vetor_df(a_words)
#     qa_vectors[k] = pd.concat([q_vec_df, a_vec_df], axis=1)

# one hot encoding
dummies = {}
features_all = pd.concat(features.values(), axis=0)
dummies_all = pd.get_dummies(features_all)

idx = 0
k_all = fn.keys()
for i in range(len(k_all)):
    k = k_all[i]
    new_idx = idx + data[k].shape[0]
    dummies[k] = dummies_all.iloc[idx:new_idx]
    print k, idx, new_idx, dummies[k].shape, data[k].shape
    idx = new_idx


# try some supervised models using sklearn
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from learning2rank.rank import RankNet, ListNet
# reload(RankNet)
# reload(ListNet)


# sklearn models
models = {}

# RandomForestClassifier
# use Genetic Algorithm to find a respectable param configuration, and then set indiv manually
# from ParamGA import *
# g = RFGA(data_trn.QuestionID, dummies['trn'], data_trn.Label, data_dev.QuestionID, dummies['dev'], data_dev.Label, pair_wise=False)
# g.run(NPOP=30, NGEN=20, CXPB=0.5, MUTPB=0.5)
indiv = [1560, 'entropy', 3, 5, 5]
models['rf'] = model_rf = RandomForestClassifier(n_estimators=indiv[0], criterion=indiv[1],
                                                 max_depth=indiv[2], min_samples_split=indiv[3],
                                                 min_samples_leaf=indiv[4], n_jobs=-1) #

# AdaBoostClassifier
#models['ada'] = model_ada = AdaBoostClassifier(base_estimator=None, n_estimators=100, learning_rate=.5, algorithm='SAMME.R', random_state=None)

# LogisticRegression
models['lr_l1'] = model_lr_l1 = LogisticRegression(penalty='l1', n_jobs=-1)
models['lr_l2'] = model_lr_l2 = LogisticRegression(penalty='l2', n_jobs=-1)

#models['svm'] = model_svm = SVC(probability=True)

# try models using learning2rank
rk_models = {}
rk_models['ranknet'] = rk_model_ranknet = RankNet.RankNet(silent=True)
#rk_models['listnet'] = rk_model_listnet = ListNet.ListNet(silent=True)

# try models using self-implemented pairwise ranker
my_models = {}
pair_ranker = PairWiseRanker(data_trn.QuestionID, dummies['trn'], data_trn.Label)
indiv =[327, 'gini', 5, 5, 1]
my_models['my_rf'] = my_models_rf = RandomForestClassifier(n_estimators=indiv[0], criterion=indiv[1],
                                                 max_depth=indiv[2], min_samples_split=indiv[3],
                                                 min_samples_leaf=indiv[4], n_jobs=-1)



# fit models

# sklearn models
for mn, md in models.items():
    md.fit(dummies['trn'].values, data_trn.Label)

# learning2rank models
for mn in rk_models: #
    args = {}
    if mn == 'ranknet':
        args = {} # {'batchsize':100, 'n_iter':5000, 'n_units1':512, 'n_units2':128, 'tv_ratio':0.95}
    elif mn == 'listnet':
        args = {'batchsize':100, 'n_epoch':1, 'n_units1':32, 'n_units2':16, 'tv_ratio':0.9}
    rk_models[mn].fit(dummies['trn'].values, data_trn.Label, **args)

# self-implemented pairwise models
for mn in my_models:
    pair_ranker.fit(my_models[mn], mn)


# rank and evaluate

sfn = fn.keys() # specify a subset of fn

# predict
preds = {k:{mn:md.predict_proba(dummies[k]) for mn, md in models.items()} for k in sfn}
rk_preds = {k:{mn:md.predict(dummies[k].values) for mn, md in rk_models.items()} for k in sfn}
my_preds = {k:{mn:pair_ranker.predict(data[k]['QuestionID'], dummies[k], mn) for mn in my_models} for k in sfn}

# rank
ranks = {k:deepcopy(dummies[k]) for k in fn}
dummy_rank_features = {k:deepcopy(dummies[k]) for k in fn}
rank_trn, rank_dev, rank_test = ranks['trn'], ranks['dev'], ranks['test']
all_tps = tps + tps_wgt + tps_wv + tps_wv_wgt + models.keys() + rk_models.keys() + my_models.keys()
for k in sfn:
    rk = ranks[k]
    for col in ['QuestionID', 'SentenceID']:
        rk[col] = data[k][col]
    if k != 'test':
        rk['Label'] = data[k]['Label']
    for mn in models:
        rk[mn] = preds[k][mn][:, 1] # larger is better
    for mn in rk_models:
        rk[mn] = rk_preds[k][mn][:, 0]
    for mn in my_models:
        rk[mn] = my_preds[k][mn]
    grp = rk.groupby('QuestionID')
    for tp in all_tps:
        col = 'rank_' + tp
        rk[col] = grp[tp].rank(method='min', ascending=False).astype(int)
        rk.sort_values(['QuestionID', col], inplace=True)
        # add one Strictly Increasing Monotonically vector to avoid same ranks
        b = np.ones(rk.shape[0]).cumsum()
        rk[col] += b
        grp = rk.groupby('QuestionID')
        rk[col] = grp[col].rank(method='min', ascending=True).astype(int)
        dummy_rank_features[k][col] = rk[col]

# evaluate using self-implemented MAP and MRR
for k in sfn:
    if k in ['test', 'trn']:
        continue
    print k
    rk = ranks[k]
    for tp in all_tps:
        col = 'rank_' + tp
        rk.sort_values(['QuestionID', col], inplace=True)
        grp = rk.Label.groupby(rk.QuestionID)
        print '\t', '{: <20}'.format(tp), np.round([MAP(grp), MRR(grp), MAP(grp, keep_no_ans=False), MRR(grp, keep_no_ans=False)], 5)


# evaluate using eval.py
# print sfn, '\n'
# print ' ' * 15, '{: >15}\t{: >15}'.format('MAP', 'MRR') * 2
# for tp in tps + tps_wgt + models.keys() + rk_models.keys() + my_models.keys(): # ['rf']: #
#     print '{: >20}'.format(tp),
#     for k in sfn:
#         if k in ['test', 'trn']:
#             continue
#         col = 'rank_' + tp
#         #k, col = 'testx', 'rank_' + 'rf'
#         ranks[k][['QuestionID', 'SentenceID', col]].sort_values(['QuestionID', col]).to_csv('rank.txt', sep='\t', header=False, index=False)
#         #print '\n', tp
#         #!python eval.py rank.txt data/WikiQA-dev.tsv
#         out = commands.getoutput('python eval.py rank.txt %s' % fn[k])
#         m = re.search('MAP: ([.0-9]+)\nMRR ([.0-9]+)', out)
#         print '\t%f\t%f' % (float(m.group(1)), float(m.group(2))),' ',
#     print ''


# define some functions

def rank_and_evaluate(y, k):
    '''
    if k is 'test', then this func will only rank, not evaluate
    '''
    d = DataFrame({}, data[k].index)
    d['y'] = y
    cols = ['QuestionID', 'SentenceID']
    if k != 'test':
        cols.append('Label')
    for col in cols:
        d[col] = data[k][col]
    grp = d.groupby('QuestionID')
    d['rank'] = grp['y'].rank(method='min', ascending=False).astype(int)

    d.sort_values(['QuestionID', 'rank'], inplace=True)
    res = dict()
    res['rank_df'] = d
    if k == 'test':
        return res
    grp = d.Label.groupby(d.QuestionID)
    res['my_MAP'] = MAP(grp, keep_no_ans=False)
    res['my_MRR'] = MRR(grp, keep_no_ans=False)
    d[['QuestionID', 'SentenceID', 'rank']].sort_values(['QuestionID', 'rank']).to_csv('rank.txt', sep='\t', header=False, index=False)
    out = commands.getoutput('python eval.py rank.txt %s' % fn[k])
    m = re.search('MAP: ([.0-9]+)\nMRR ([.0-9]+)', out)
    res['MAP'] = float(m.group(1))
    res['MRR'] = float(m.group(2))
    return res

def save_final_rank(df, dst_fn='final_rank.txt', rank_col='rank',
                    save_cols=['QuestionID', 'SentenceID', 'rank'], sort_by_cols=['QuestionID', 'rank'], re_rank=False):
    if re_rank:
        col = rank_col
        df.sort_values(['QuestionID', col], inplace=True)
        # add one Strictly Increasing Monotonically vector to avoid same ranks
        b = np.ones(df.shape[0]).cumsum()
        df[col] += b
        grp = df.groupby('QuestionID')
        df[col] = grp[col].rank(method='min', ascending=True).astype(int)
    df[save_cols].sort_values(sort_by_cols).to_csv(dst_fn, sep='\t', header=False, index=False)

def resample(rt, Xs, Y):
    if rt == None or rt == 0:
        return Xs, Y
    n = Y.shape[0]
    n1 = Y.sum()
    n0 = n - n1
    cls_idx = np.arange(n)
    pos_cls_idx = cls_idx[Y==1]
    neg_cls_idx = cls_idx[Y==0]
    pos_sample = np.random.choice(pos_cls_idx, size=int(n0 / float(rt)) - n1, )
    new_cls_idx = np.concatenate([neg_cls_idx, pos_cls_idx, pos_sample])
    np.random.shuffle(new_cls_idx)
    print n0, n1, n, int(n0 / float(rt)) - n1, new_cls_idx.shape
    new_Y = Y[new_cls_idx]
    new_Xs = [X[new_cls_idx] for X in Xs]
    return new_Xs, new_Y

# try neural networks

# print('Found %s unique tokens.' % len(word_index))
# print max([len(seq) for seq in ques_seq.values()]), max([len(seq) for seq in sent_seq.values()])
# MAX_SENT_SEQUENCE_LENGTH = max([max([len(seq) for seq in sents_seq[k]]) for k in fn])
ques_len = Series([len(q) for q in ques_seq.values()])
#ques_len.hist()
sent_len = Series([len(s) for s in sent_seq.values()])
#sent_len.hist()
#sent_len[sent_len<200].hist()

# pad questions and sentences
MAX_QUES_SEQUENCE_LENGTH, MAX_SENT_SEQUENCE_LENGTH = 20, 60
from keras.preprocessing.sequence import pad_sequences
sent_pad = dict(zip(sent_seq.keys(), pad_sequences(sent_seq.values(), maxlen=MAX_SENT_SEQUENCE_LENGTH, padding='post', truncating='post')))
ques_pad = dict(zip(ques_seq.keys(), pad_sequences(ques_seq.values(), maxlen=MAX_QUES_SEQUENCE_LENGTH, padding='post', truncating='post')))
# pad questions as sentences
ques_as_sent_pad = dict(zip(ques_seq.keys(), pad_sequences(ques_seq.values(), maxlen=MAX_SENT_SEQUENCE_LENGTH, padding='post', truncating='post')))

# convert to DataFrame
sent_pad_df = DataFrame(sent_pad.values(), index=sent_pad.keys())
ques_pad_df = DataFrame(ques_pad.values(), index=ques_pad.keys())
ques_as_sent_pad_df = DataFrame(ques_as_sent_pad.values(), index=ques_as_sent_pad.keys())

paddata = collections.defaultdict(dict)
for k in fn:
    paddata[k] = {'ques':ques_pad_df.loc[data[k]['QuestionID']],
                  'ques_as_sent':ques_as_sent_pad_df.loc[data[k]['QuestionID']],
                  'sent':sent_pad_df.loc[data[k]['QAID']]}

num_output_class = 1
label = {}
for k in ['trn', 'dev', 'testx']:
    tmp = data[k]['Label'].values
    if num_output_class == 1:
        label[k] = tmp
    elif num_output_class == 2:
        label[k] = np.array([1 - tmp, tmp]).T


# try adapted CNN

# first, config something
use_add_features = True # use additional features?
num_output_class = 1
features_used = dummy_rank_features
add_features_dim = features_used['trn'].shape[1] if use_add_features else None
X, Y = {}, {}
for k in sfn:
    X[k] = [paddata[k]['ques'].values, paddata[k]['sent'].values]
    if k != 'test':
        Y[k] = label[k]
    if use_add_features:
        X[k].append(features_used[k].values)

from MyCNN import *
cnn = MyCNN(embedding_matrix, word_index, MAX_QUES_SEQUENCE_LENGTH, MAX_SENT_SEQUENCE_LENGTH, add_features_dim)
cnn.init_model(num_filters=100, filter_size=5, pool_length=2,
               denses=[100], dropouts=[0], activations=['relu'],
               num_output_class=num_output_class)

cnn.fit(X['trn'], Y['trn'], validation_data=(X['dev'], Y['dev']), nb_epoch=2, batch_size=50) # data['trn'].shape[0]) #

# evaluate CNN
res = {}
for k in sfn:
    if k in ['trn', 'test']:
        continue
    y = cnn.predict(X[k])
    res[k] = rank_and_evaluate(y[:, 0], k)
    print k
    for m in ['my_MAP', 'my_MRR', 'MAP', 'MRR']:
        print m, '\t', res[k][m]

# save model
# model.save('model/cnn_2d_%d_%f_%f.h5' % (add_features_dim, res['dev']['MRR'], res['testx']['MRR']))


# try adapted LSTM

# first, configurate something. prepare X and Y
use_add_features = True
features_used = dummy_rank_features
add_features_dim = features_used['trn'].shape[1] if use_add_features else None
X, Y = {}, {}
for k in sfn:
    X[k] = [paddata[k]['ques_as_sent'].values, paddata[k]['sent'].values]
    if k != 'test':
        Y[k] = label[k]
    if use_add_features:
        X[k].append(features_used[k].values)

from MyLSTM import *
model = MyLSTM(embedding_matrix, word_index, MAX_SENT_SEQUENCE_LENGTH, add_features_dim)
model.init_model(lstm_output_dim=64, denses=[64], dropouts=[0])
#np.random.seed(1337) # not work
model.fit(X['trn'], Y['trn'], validation_data=(X['dev'], Y['dev']), nb_epoch=2, batch_size=64)

# evaluate adapted LSTM
res = {}
for k in sfn:
    if k in ['trn', 'test']:
        continue
    y = model.predict(X[k])
    res['y'] = y
    res[k] = rank_and_evaluate(y, k)
    print k
    for m in ['my_MAP', 'my_MRR', 'MAP', 'MRR']:
        print m, '\t', res[k][m]

# predict on test dataset
k = 'test'
y = model.predict(X[k])
tres = rank_and_evaluate(y, k)
save_final_rank(tres['rank_df'], dst_fn='final_rank.txt', rank_col='rank',
                    save_cols=['QuestionID', 'SentenceID', 'rank'], sort_by_cols=['QuestionID', 'rank'], re_rank=False)

# save model
# model.save('model/lstm_2d_%d_%f_%f.h5' % (add_features_dim, res['dev']['MRR'], res['testx']['MRR']))

# from keras.models import load_model
# model = load_model('model/lstm.1337.709692.663293.h5')

# predict and save
# predict on test dataset
k = 'test'
y = model.predict(X[k])
tres = rank_and_evaluate(y, k)
save_final_rank(tres['rank_df'], dst_fn='final_rank.txt', rank_col='rank',
                    save_cols=['QuestionID', 'SentenceID', 'rank'], sort_by_cols=['QuestionID', 'rank'], re_rank=False)

print 'Done'
