import cv2
import mediapipe as mp
import numpy as np
import math
import tensorflow as tf
import pickle

# Configuración de MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Configuración del teclado
teclas = [
    ["1","2","3","4","5","6","7","8","9","0"],
    ["Q","W","E","R","T","Y","U","I","O","P"],
    ["A","S","D","F","G","H","J","K","L","Ñ"],
    ["Z","X","C","V","B","N","M",",",".","-"],
    ["TAB","SPACE","DEL","!","?"]
]
tecla_ancho = 50
tecla_alto = 50
texto_escrito = ""

# Parámetro de entrada del modelo
SEQ_LENGTH = 3

# Cargar modelo y tokenizer
model = tf.keras.models.load_model('model/autocomplete_es.h5')
with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Función para detectar toque entre pulgar e índice
def detectar_toque(landmarks, width, height):
    x1 = int(landmarks[8].x * width)
    y1 = int(landmarks[8].y * height)
    x2 = int(landmarks[4].x * width)
    y2 = int(landmarks[4].y * height)
    distancia = math.hypot(x2 - x1, y2 - y1)
    if distancia < 40:
        return x1, y1
    return None, None

# Función de predicción de autocompletado
def predecir_palabras(texto, max_sugerencias=3):
    palabras = texto.lower().split()
    contexto = palabras[-SEQ_LENGTH:] if len(palabras) >= SEQ_LENGTH else [''] * (SEQ_LENGTH - len(palabras)) + palabras
    secuencia = tokenizer.texts_to_sequences([' '.join(contexto)])
    if not secuencia or len(secuencia[0]) == 0:
        return []
    secuencia = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=SEQ_LENGTH)
    predicciones = model.predict(secuencia)[0]
    indices_ordenados = predicciones.argsort()[-max_sugerencias:][::-1]
    
    sugerencias = []
    idx_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    for idx in indices_ordenados:
        palabra = idx_to_word.get(idx)
        if palabra and palabra not in sugerencias:
            sugerencias.append(palabra)
    return sugerencias

# Función para dibujar teclado, procesar entrada y mostrar sugerencias
def dibujar_teclado(img, texto_escrito, punto_toque, sugerencias):
    global teclas
    tecla_presionada = None
    y0 = 60
    for fila in teclas:
        x0 = 40
        for tecla in fila:
            ancho = tecla_ancho
            if tecla == "SPACE":
                ancho = tecla_ancho * 4
            elif tecla in ["TAB", "DEL"]:
                ancho = tecla_ancho * 2

            if punto_toque:
                x, y = punto_toque
                if x0 < x < x0 + ancho and y0 < y < y0 + tecla_alto:
                    cv2.rectangle(img, (x0, y0), (x0+ancho, y0+tecla_alto), (0,255,0), -1)
                    tecla_presionada = tecla
                else:
                    cv2.rectangle(img, (x0, y0), (x0+ancho, y0+tecla_alto), (255,0,0), 2)
            else:
                cv2.rectangle(img, (x0, y0), (x0+ancho, y0+tecla_alto), (255,0,0), 2)

            font_scale = 0.6 if len(tecla) == 1 else 0.5
            cv2.putText(img, tecla, (x0+5, y0+35), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255,255,255), 1)
            x0 += ancho + 5
        y0 += tecla_alto + 5

    # Dibuja sugerencias como "botones" debajo del teclado
    sugerencia_y = y0 + 20
    x0 = 40
    sugerencia_ancho = 150
    sugerencia_alto = 40
    tecla_sugerida = None
    if punto_toque:
        x, y = punto_toque
    else:
        x, y = None, None
    for i, sugerencia in enumerate(sugerencias):
        # Detectar toque en las sugerencias
        if x and y and x0 < x < x0 + sugerencia_ancho and sugerencia_y < y < sugerencia_y + sugerencia_alto:
            cv2.rectangle(img, (x0, sugerencia_y), (x0+sugerencia_ancho, sugerencia_y+sugerencia_alto), (0,255,0), -1)
            tecla_sugerida = sugerencia
        else:
            cv2.rectangle(img, (x0, sugerencia_y), (x0+sugerencia_ancho, sugerencia_y+sugerencia_alto), (100,100,100), 2)
        cv2.putText(img, sugerencia, (x0+10, sugerencia_y+27), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        x0 += sugerencia_ancho + 10

    # Procesar tecla normal
    if tecla_presionada:
        if tecla_presionada == "SPACE":
            texto_escrito += " "
        elif tecla_presionada == "TAB":
            texto_escrito += "    "
        elif tecla_presionada == "DEL":
            texto_escrito = texto_escrito[:-1]
        else:
            texto_escrito += tecla_presionada

    # Procesar tecla sugerida (autocompletar)
    if tecla_sugerida:
        # Agregar la palabra sugerida al texto sin eliminar nada
        texto_escrito += tecla_sugerida + ' '

    return img, texto_escrito

# Captura de video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.6) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        height, width, _ = frame.shape
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        punto_toque = None
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0,0,0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=2)
                )
                x, y = detectar_toque(hand_landmarks.landmark, width, height)
                if x and y:
                    punto_toque = (x, y)

        sugerencias = predecir_palabras(texto_escrito)

        frame, texto_escrito = dibujar_teclado(frame, texto_escrito, punto_toque, sugerencias)

        # Mostrar texto
        cv2.rectangle(frame, (40, 10), (1240, 50), (0,0,0), -1)
        cv2.putText(frame, texto_escrito, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

        # Mostrar sugerencias como texto adicional (opcional)
        # (Ya están en botones, pero también puedes mostrarlo aquí si quieres)
        # if sugerencias:
        #     cv2.putText(frame, "Sugerencias: " + ', '.join(sugerencias), (50, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Teclado Virtual", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
