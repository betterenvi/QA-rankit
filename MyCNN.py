from keras.layers import Dense, Input, Flatten, Dropout, Embedding, LSTM, Merge, merge
from keras.layers import Conv1D, MaxPooling1D, Embedding, Merge, merge
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras.backend as K
import sklearn

class MyCNN(object):
    def __init__(self, embedding_matrix, word_index, MAX_QUES_SEQUENCE_LENGTH, MAX_SENT_SEQUENCE_LENGTH, add_features_dim):
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = embedding_matrix.shape[1]
        self.word_index = word_index
        self.add_features_dim = add_features_dim
        self.MAX_QUES_SEQUENCE_LENGTH = MAX_QUES_SEQUENCE_LENGTH
        self.MAX_SENT_SEQUENCE_LENGTH = MAX_SENT_SEQUENCE_LENGTH
        
    def _gen_layers(self, num_filters, filter_size, pool_length, MAX_SEQUENCE_LENGTH, transform=False, transform_dim=100):
        embedding_layer = Embedding(len(self.word_index) + 1,
                                self.EMBEDDING_DIM,
                                weights=[self.embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
        sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        conv = Conv1D(num_filters, filter_size, border_mode='same', activation='relu')(embedded_sequences)
        pooling = MaxPooling1D(pool_length)(conv)
        flatten_pooling = Flatten()(pooling)
        if transform:
            transformed = Dense(transform_dim, activation='linear')(flatten_pooling)
            #transformed = Dropout(0.5)(transformed)
        else:
            transformed = flatten_pooling
        seq_model = Model(sequence_input, transformed)
        return sequence_input, flatten_pooling, transformed, seq_model

    def init_model(self, num_filters=100, filter_size=5, pool_length=1, 
                  denses=[], dropouts=[], activations=[], num_output_class=1):
        ques_input, ques_flatten_pooling, ques_transformed, ques_model = self._gen_layers(
            num_filters, filter_size, pool_length, self.MAX_QUES_SEQUENCE_LENGTH, transform=False)
        sent_input, sent_flatten_pooling, sent_transformed, sent_model = self._gen_layers(
            num_filters, filter_size, pool_length, self.MAX_SENT_SEQUENCE_LENGTH, transform=True, transform_dim=100)
        merged_input = [ques_input, sent_input]
        merged_layer_list = [ques_transformed, sent_transformed] 
        if self.add_features_dim != None:
            add_features_input = Input(shape=(self.add_features_dim,))
            merged_input.append(add_features_input)
            merged_layer_list.append(add_features_input)
        merged = merge(merged_layer_list, mode='concat', concat_axis=1)
        for i, d in enumerate(denses):
            merged = Dense(d, activation=activations[i])(merged)
            merged = Dropout(dropouts[i])(merged)
        if num_output_class == 1:
            output = Dense(1, activation='sigmoid', name='main_output')(merged)
            model = Model(input=merged_input, output=output)
            model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        else:
            output = Dense(2, activation='softmax')(merged)
            model = Model(input=merged_input, output=output)
            model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model = model
        
    def fit(self, X, Y, *args, **kwargs):
        self.model.fit(X, Y, *args, **kwargs)
    
    def predict(self, X):
        self.preds = self.model.predict(X)
        return self.preds

#     def predict_proba(self, X):
#         self.pred_probas = self.model.predict_proba(X)
#         return self.pred_probas
    
    def save(self, fn):
        self.model.save(fn)
        