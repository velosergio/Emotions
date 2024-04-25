from deepface import DeepFace

# Analizar la imagen "img1.jpg"
results = DeepFace.analyze(img_path="img\img1.jpg", actions=['emotion'])

# Verificar si se detectaron caras en la imagen
if len(results) > 0:
    # Obtener el primer resultado de análisis
    first_result = results[0]
    
    # Obtener la emoción predominante del primer resultado
    emotion = first_result['dominant_emotion']
    
    # Mostrar la emoción predominante en la consola
    print("La emoción predominante en la imagen es:", emotion)
else:
    print("No se detectaron caras en la imagen")
