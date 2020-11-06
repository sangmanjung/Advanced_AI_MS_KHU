from konlpy.tag import Komoran
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import numpy as np
import argparse

# argument
parser = argparse.ArgumentParser()
parser.add_argument('--train_file',type=str)
parser.add_argument('--input_file',type=str)
parser.add_argument('--output_file',type=str)
args = parser.parse_args()
    
# Tokenizer for encoding
def tokenize(samples):
      tokenizer = Tokenizer()
      tokenizer.fit_on_texts(samples)
      return tokenizer

# Data load
data = pd.read_csv(args.input_file,names=['Text','Label'], sep = '\s+',quoting = 3)
data.head()

# toList
data_value = data.Text.values
data_to_list = list(data_value)

# Preprocessing
komoran = Komoran()
texts_test = []
for i in range(len(data_to_list)):
    postagging = komoran.morphs(data_to_list[i])
    texts_test.append(postagging)

tx_token = tokenize(texts_test)

x_test = tx_token.texts_to_sequences(texts_test)

max_len = 10 # according to the histogram
x_test = pad_sequences(x_test, padding = 'post', maxlen = max_len)

# load the model
from tensorflow.keras.models import load_model
model = load_model('model.h5')

# index sets
index_to_word = tx_token.index_word
index_to_tag = {}
with open('train_tag.txt') as file:
    for line in file:
        (key, val) = line.split()
        index_to_tag[int(key)] = val

# prediction
y_hat = model.predict(x_test)
y_hat = np.argmax(y_hat,-1)

# save the result as the text file : result.txt
out = open(args.output_file,'w')
outputlist = []
for i in range(len(x_test)):
    outputlist.clear()
    for w, p in zip(x_test[i],y_hat[i]):
        if w != 0: # except 'pad'
            outputlist.append(index_to_word[w])
            outputlist.append('/' + index_to_tag[p].upper()+'+')
    tmpstring = ''.join(outputlist)
    print(tmpstring[:-1],file = out)
out.close()