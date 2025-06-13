# Teclado Virtual con Autocompletado en Español

Este proyecto muestra un prototipo de teclado virtual controlado con los dedos mediante visión por computadora, sonido al presionar teclas y autocompletado de palabras en español usando redes neuronales (LSTM).

## 📁 Estructura

- `model/`: contiene el modelo LSTM entrenado y el tokenizador.
- `sounds/`: sonido de tecla (`tecla.wav`).
- `src/`: archivos Python del prototipo y el entrenamiento.

## ⚙️ Requisitos

- Python 3.7 o superior
- Librerías:
  ```bash
  pip install opencv-python mediapipe pygame numpy tensorflow
