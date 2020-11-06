from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, InputLayer, Bidirectional, TimeDistributed, Embedding, Dropout
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Embedding(vocab_size, 128, input_length = max_len, mask_zero = True))
model.add(Bidirectional(LSTM(256, return_sequences = True)))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(tag_size, activation = ('relu'))))
model.add(TimeDistributed(Dense(tag_size, activation = ('softmax'))))
model.compile(loss = 'categorical_crossentropy',optimizer = Adam(0.001),metrics = ['accuracy'])