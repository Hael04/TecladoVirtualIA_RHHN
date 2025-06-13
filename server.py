# --- 2) server.py ---
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import pickle

app = Flask(__name__)
SEQ_LENGTH = 3

# Cargar modelo y tokenizer
model = tf.keras.models.load_model('model/autocomplete_es.h5')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '').lower().split()
    # Tomar últimas SEQ_LENGTH palabras
    context = text[-SEQ_LENGTH:] if len(text) >= SEQ_LENGTH else ['']*(SEQ_LENGTH-len(text)) + text
    seq = tokenizer.texts_to_sequences([' '.join(context)])[0]
    seq = np.array([seq])
    preds = model.predict(seq)[0]
    next_idx = np.argmax(preds)
    # Obtener palabra
    for w, i in tokenizer.word_index.items():
        if i == next_idx:
            next_word = w
            break
    return jsonify({'suggestion': next_word})

if __name__ == '__main__':
    app.run(debug=True)