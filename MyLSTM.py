from keras.layers import Dense, Input, Flatten, Dropout, Embedding, LSTM, Merge, merge
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import keras.backend as K

class MyLSTM(object):
    
    def __init__(self, embedding_matrix, word_index, MAX_SENT_SEQUENCE_LENGTH, add_features_dim):
        self.embedding_matrix = embedding_matrix
        self.EMBEDDING_DIM = embedding_matrix.shape[1]
        self.word_index = word_index
        self.MAX_SENT_SEQUENCE_LENGTH = MAX_SENT_SEQUENCE_LENGTH
        self.add_features_dim = add_features_dim
    
    def _gen_embedding_layer(self):
        embedding_layer = Embedding(len(self.word_index) + 1, 
                                    self.EMBEDDING_DIM,
                                    weights=[self.embedding_matrix],
                                    input_length=self.MAX_SENT_SEQUENCE_LENGTH,
                                    trainable=False)
        sequence_input = Input(shape=(self.MAX_SENT_SEQUENCE_LENGTH,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        return sequence_input, embedded_sequences
    
    def init_model(self, lstm_output_dim=64, denses=[], dropouts=[]):
        ques_input, ques_embedded = self._gen_embedding_layer()
        sent_input, sent_embedded = self._gen_embedding_layer()
        shared_lstm = LSTM(lstm_output_dim,dropout_U=0)
        ques_encoded = shared_lstm(ques_embedded)
        sent_encoded = shared_lstm(sent_embedded)
        merged_layer_list = [ques_encoded, sent_encoded]
        merged_input = [ques_input, sent_input]
        if self.add_features_dim != None:
            add_features_input = Input(shape=(self.add_features_dim,))
            merged_layer_list.append(add_features_input)
            merged_input.append(add_features_input)
        merged_vector = merge(merged_layer_list, mode='concat', concat_axis=-1, name='lstm_vec')
        for i, d in enumerate(denses):
            merged_vector = Dense(d,activation='sigmoid')(merged_vector)
            merged_vector = Dropout(dropouts[i])(merged_vector)
        predictions = Dense(1, activation='sigmoid')(merged_vector)
        self.model = Model(input=merged_input, output=predictions)
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        
    def init_model_v2(self, lstm_output_dim=64, denses=[], dropouts=[]):
        ques_input, ques_embedded = self._gen_embedding_layer()
        sent_input, sent_embedded = self._gen_embedding_layer()
        shared_lstm = LSTM(lstm_output_dim,dropout_U=0)
        ques_encoded = shared_lstm(ques_embedded)
        sent_encoded = shared_lstm(sent_embedded)
        merged_layer_list = [ques_encoded, sent_encoded]
        merged_input = [ques_input, sent_input]
        merged_vector = merge(merged_layer_list, mode='concat', concat_axis=-1, name='lstm_vec')
        for i, d in enumerate(denses):
            merged_vector = Dense(d,activation='sigmoid')(merged_vector)
            merged_vector = Dropout(dropouts[i])(merged_vector)
        if self.add_features_dim != None:
            add_features_input = Input(shape=(self.add_features_dim,))
            merged_vector = merge([merged_vector, add_features_input], mode='concat', concat_axis=-1, name='vec')
            merged_input.append(add_features_input)
        predictions = Dense(1, activation='sigmoid')(merged_vector)
        self.model = Model(input=merged_input, output=predictions)
        self.model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])
        
    def fit(self, X, Y, *args, **kwargs):
        self.model.fit(X, Y, *args, **kwargs)
    
    def predict(self, X):
        self.preds = self.model.predict(X)
        return self.preds
    
    def save(self, fn):

        self.model.save(fn)