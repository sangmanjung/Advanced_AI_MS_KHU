import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--train_file',type=str)
parser.add_argument('--input_file',type=str)
parser.add_argument('--output_file',type=str)
args = parser.parse_args()
    
# tokenizer
def tokenize(samples):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(samples)
      return tokenizer

# data load
data = pd.read_csv(args.train_file,names=['Text','Label'],sep='\s+',quoting=3)
new_data = pd.concat([data.Label.str.split('+',expand = True)], axis = 1)

# change to list
a = pd.Series.tolist(new_data[0:-1])
tmp = []

# data split
for i in range(len(a)):
    d = []
    for j in range(len(a[0])):
        if a[i][j] != None:
            b = a[i][j].split('/')
            b = tuple(b)
            d.append(b)
    tmp.append(d)

# consider '/'
for i in range(len(tmp)):
    for j in range(len(tmp[i])):
        if tmp[i][j] == ('', '', 'SP'):
            tmp[i][j] = ('/','SP')

# obtain the text and pos tags (separation)		
texts, pos_tags = [],[] 
for t in tmp:
    text, tag = zip(*t)
    texts.append(list(text))
    pos_tags.append(list(tag))

# tokenizing
tx_token = tokenize(texts)
pos_token = tokenize(pos_tags)

with open('train_tag.txt', 'w') as file:
    for key, item in zip(pos_token.index_word.keys(),pos_token.index_word.values()):
        print(key,' ',item, file = file)

# size of vocab & tag set
vocab_size = len(tx_token.word_index) + 1
tag_size = len(pos_token.word_index) + 1

# change to the sequence
x_train = tx_token.texts_to_sequences(texts)
y_train = pos_token.texts_to_sequences(pos_tags)

# zero padding
max_len = 10 # according to the histogram
x_train = pad_sequences(x_train, padding = 'post', maxlen = max_len)
y_train = pad_sequences(y_train, padding = 'post', maxlen = max_len)

# one-hot encoding
y_train = to_categorical(y_train, num_classes = tag_size)

# LSTMmodel
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_len, mask_zero = True))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(tag_size, activation = ('relu'))))
model.add(TimeDistributed(Dense(tag_size, activation = ('softmax'))))
model.compile(loss = 'categorical_crossentropy',optimizer = Adam(0.001),metrics = ['accuracy'])

# model training
model.fit(x_train, y_train, batch_size = 10, epochs = 2)

model.save('model.h5')