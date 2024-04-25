import cv2
import numpy as np
from deepface import DeepFace
import tempfile
import os

# Crear una instancia del clasificador de rostros de OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Mapeo de emociones en inglés a español
emotions_mapping = {
    'angry': 'enojado',
    'disgust': 'asco',
    'fear': 'miedo',
    'happy': 'feliz',
    'sad': 'triste',
    'surprise': 'sorpresa',
    'neutral': 'neutral'
}

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

while True:
    # Leer un cuadro del flujo de video de la cámara web
    ret, frame = cap.read()
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Verificar si se detectaron caras en la imagen
    if len(faces) > 0:
        # Para cada rostro detectado, analizar la emoción predominante, el género y la edad
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) que contiene el rostro
            roi_gray = gray[y:y+h, x:x+w]
            
            # Crear un archivo temporal para almacenar la ROI como una imagen
            temp_img_path = os.path.join(tempfile.gettempdir(), "temp_img.jpg")
            
            # Guardar la ROI como una imagen temporal
            cv2.imwrite(temp_img_path, roi_gray)
            
            # Analizar la emoción, el género y la edad en la imagen temporal
            results = DeepFace.analyze(img_path=temp_img_path, actions=['emotion', 'gender', 'age'], enforce_detection=False)
            
            # Eliminar el archivo temporal después de usarlo
            os.remove(temp_img_path)
            
            # Verificar si se detectaron emociones, género y edad en la ROI
            if len(results) > 0:
                # Obtener la emoción predominante, el género y la edad del primer resultado
                emotion = results[0]['dominant_emotion']
                gender = results[0]['gender']
                age = results[0]['age']
                
                # Traducir la emoción y el género a español
                emotion_spanish = emotions_mapping.get(emotion, emotion)
                gender_spanish = "hombre" if gender == 'Man' else "mujer"
                
                # Dibujar un rectángulo alrededor del rostro
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Mostrar la emoción, el género y la edad encima del rectángulo
                text = f"{emotion_spanish}, {gender_spanish}, {int(age)} años"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar el cuadro de video con las emociones, género y edad detectadas
    cv2.imshow('Detección de Emociones', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
