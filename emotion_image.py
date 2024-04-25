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

# Función para cargar la imagen de "mona_{emotion}"
def load_mona_image(emotion):
    img_path = os.path.join("img", f"mona_{emotion.lower()}.jpg")
    if os.path.exists(img_path):
        return cv2.imread(img_path)
    else:
        print(f"Error: No se pudo cargar la imagen {img_path}")
        return None

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

# Crear una ventana para mostrar la cámara web
cv2.namedWindow('Detección de Emociones', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Detección de Emociones', 712, 801)

# Crear una ventana para mostrar la imagen de "mona_neutral"
cv2.namedWindow('Mona Lisa', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Mona Lisa', 712, 801)

# Cargar la imagen de "mona_neutral"
mona_neutral = load_mona_image("neutral")

while True:
    # Leer un cuadro del flujo de video de la cámara web
    ret, frame = cap.read()
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detectar rostros en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Verificar si se detectaron caras en la imagen
    if len(faces) > 0:
        # Para cada rostro detectado, analizar la emoción predominante
        for (x, y, w, h) in faces:
            # Extraer la región de interés (ROI) que contiene el rostro
            roi_gray = gray[y:y+h, x:x+w]
            
            # Crear un archivo temporal para almacenar la ROI como una imagen
            temp_img_path = os.path.join(tempfile.gettempdir(), "temp_img.jpg")
            
            # Guardar la ROI como una imagen temporal
            cv2.imwrite(temp_img_path, roi_gray)
            
            # Analizar la emoción predominante en la imagen temporal
            results = DeepFace.analyze(img_path=temp_img_path, actions=['emotion'], enforce_detection=False)
            
            # Eliminar el archivo temporal después de usarlo
            os.remove(temp_img_path)
            
            # Verificar si se detectaron emociones en la ROI
            if len(results) > 0:
                # Obtener la emoción predominante del primer resultado
                emotion = results[0]['dominant_emotion']
                
                # Traducir la emoción a español
                emotion_spanish = emotions_mapping.get(emotion, emotion)
                
                # Cargar la imagen de "mona_{emotion}"
                mona_image = load_mona_image(emotion)
                
                # Verificar si la imagen se cargó correctamente
                if mona_image is not None:
                    # Redimensionar la imagen de "mona_{emotion}" para que coincida con el tamaño de la ventana
                    mona_image = cv2.resize(mona_image, (712, 801))
                    
                    # Mostrar la imagen de "mona_{emotion}" en la ventana 'Mona Lisa'
                    cv2.imshow('Mona Lisa', mona_image)
                
                # Dibujar un rectángulo alrededor del rostro en la ventana 'Detección de Emociones'
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Mostrar la emoción predominante encima del rectángulo en la ventana 'Detección de Emociones'
                cv2.putText(frame, emotion_spanish, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Mostrar el cuadro de video con las emociones detectadas en la ventana 'Detección de Emociones'
    cv2.imshow('Detección de Emociones', frame)
    
    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()
