# --- 1) train_model.py ---
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import pickle

# Parámetros
SEQ_LENGTH = 3     # Número de palabras de contexto
EMBED_DIM = 50
LSTM_UNITS = 100
BATCH_SIZE = 128
EPOCHS = 50

# 1. Cargar el corpus
with open('data/spanish_corpus.txt', encoding='utf-8') as f:
    text = f.read().lower()
# Tokenizar por palabras
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
words = tokenizer.texts_to_sequences([text])[0]
vocab_size = len(tokenizer.word_index) + 1

# 2. Crear secuencias (X: contexto, y: siguiente palabra)
sequences = []
for i in range(SEQ_LENGTH, len(words)):
    seq = words[i-SEQ_LENGTH:i+1]
    sequences.append(seq)
sequences = np.array(sequences)
X, y = sequences[:, :-1], sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# 3. Pad (aunque deberían tener tamaño constante)
# X = pad_sequences(X, maxlen=SEQ_LENGTH, padding='pre')

# 4. Definir modelo
model = Sequential([
    Embedding(vocab_size, EMBED_DIM, input_length=SEQ_LENGTH),
    LSTM(LSTM_UNITS),
    Dense(vocab_size, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# 5. Entrenar
model.fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

# 6. Guardar modelo y tokenizer
model.save('model/autocomplete_es.h5')
with open('model/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

print('Entrenamiento completado.')