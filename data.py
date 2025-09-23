import cv2
import numpy as np
import os
import time

PIXELES_POR_MM = 3.2 
MANZANA_PEQUENA_MM = 65
MANZANA_MEDIANA_MM = 80

# --- PREPARACIÓN DEL DATASET ---
# Nombres de las carpetas
CATEGORIAS = ["madura", "intermedia", "verde"]
# Crear las carpetas si no existen
for categoria in CATEGORIAS:
    path = os.path.join('dataset', categoria)
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Carpeta creada: {path}")

cap = cv2.VideoCapture(0)
print("--- Herramienta de Creacion de Dataset ---")
print("Presiona 'M' (madura), 'I' (intermedia), o 'V' (verde) para guardar una imagen.")
print("Presiona 'Q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Se dibuja un cuadro de guía en el centro para saber dónde poner la manzana
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    cv2.rectangle(frame, (cx - 150, cy - 150), (cx + 150, cy + 150), (255, 255, 255), 2)
    
    # Detección de la manzana
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    limite_bajo = np.array([0, 50, 50])
    limite_alto = np.array([180, 255, 255])
    mascara = cv2.inRange(hsv_frame, limite_bajo, limite_alto)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        contorno_manzana = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno_manzana) > 2000:
            x, y, w, h = cv2.boundingRect(contorno_manzana)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # --- LÓGICA PARA GUARDAR IMÁGENES ---
            key = cv2.waitKey(1) & 0xFF
            
            # Función para guardar
            def guardar_imagen(categoria):
                # Recortamos la manzana de la imagen original
                roi = frame[y:y+h, x:x+w]
                # Creamos un nombre de archivo único con la hora
                filename = f"img_{int(time.time() * 1000)}.png"
                filepath = os.path.join('dataset', categoria, filename)
                cv2.imwrite(filepath, roi)
                print(f"Imagen guardada en: {filepath}")

            if key == ord('m'):
                guardar_imagen("madura")
            elif key == ord('i'):
                guardar_imagen("intermedia")
            elif key == ord('v'):
                guardar_imagen("verde")
            elif key == ord('q'):
                break

    cv2.imshow('Creador de Dataset', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()