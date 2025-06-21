import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, Response, jsonify
import mediapipe as mp
from keras.models import load_model

# Garantindo a versão correta do TensorFlow
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# Inicializando o Flask
app = Flask(__name__)

# Diretório de imagens dos sinais
IMAGE_DIR = "static/sign_images"

# Inicializando o MediaPipe para detecção de mãos
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Carregando o modelo treinado
model = load_model('keras_model.h5')
classes = ['A', 'B', 'C', 'D', 'OI']
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            continue

        frameRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(frameRGB)
        handsPoints = results.multi_hand_landmarks
        h, w, _ = img.shape

        if handsPoints:
            for hand in handsPoints:
                x_max, y_max = 0, 0
                x_min, y_min = w, h

                for lm in hand.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_max, x_min = max(x_max, x), min(x_min, x)
                    y_max, y_min = max(y_max, y), min(y_min, y)

                cv2.rectangle(img, (x_min-50, y_min-50), (x_max+50, y_max+50), (0, 255, 0), 2)

                try:
                    imgCrop = img[max(0, y_min-50):min(h, y_max+50), max(0, x_min-50):min(w, x_max+50)]
                    imgCrop = cv2.resize(imgCrop, (224, 224))
                    imgArray = np.asarray(imgCrop)
                    normalized_image_array = (imgArray.astype(np.float32) / 127.0) - 1
                    data[0] = normalized_image_array
                    prediction = model.predict(data)
                    indexVal = np.argmax(prediction)

                    cv2.putText(img, classes[indexVal], (x_min-50, y_min-65), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 0, 255), 5)

                except Exception as e:
                    print(f"Erro no processamento da imagem: {e}")
                    continue

        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sign_image/<signal>')
def get_sign_image(signal):
    image_path = os.path.join(IMAGE_DIR, f"{signal}.jpg")
    if os.path.exists(image_path):
        return jsonify({"image_url": f"/{image_path}"})
    else:
        return jsonify({"error": "Imagem não encontrada"}), 404

if __name__ == '__main__':
    app.run(debug=True)