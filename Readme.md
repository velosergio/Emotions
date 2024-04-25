# Emotions

Este proyecto utiliza la biblioteca DeepFace para detectar las emociones en las caras capturadas por la cámara web en tiempo real y mostrar una imagen de Mona Lisa que refleja la emoción predominante utilizando DeepFace y openCV.

## Requerimientos

> Python 3.11

El siguiente proyecto usa las siguientes dependencias en las versiones:

1. OpenCv [4.9.0.80]
2. deepface [0.00.90]
3. tf-keras [2.16.0]

Se abrirán dos ventanas: una con el video de la cámara web y otra con una imagen de Mona Lisa que cambia según la emoción detectada.

## Personalización

Puedes personalizar el proyecto agregando nuevas imágenes de Mona Lisa para diferentes emociones. Asegúrate de seguir la convención de nomenclatura: `mona_{emotion}.jpg`, donde `{emotion}` es el nombre de la emoción en inglés (por ejemplo, `happy`, `sad`, `angry`, etc.).

## Contribución

Si encuentras algún problema o tienes sugerencias para mejorar el proyecto, ¡no dudes en abrir un nuevo problema o enviar una solicitud de extracción!

## Créditos

Este proyecto fue inspirado por [OpenCV](https://opencv.org/) y [DeepFace](https://github.com/serengil/deepface), y creado usando ChatGPT.

## Licencia

Este proyecto está bajo la [Licencia MIT](LICENSE).