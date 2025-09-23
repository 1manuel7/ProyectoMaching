import cv2
import numpy as npq

cap = cv2.VideoCapture(0)

print("Sistema de calibracion de COLOR iniciado.")
print("Coloca una manzana y anota los valores 'Hue' que aparecen en la terminal.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Rango de colores para detectar CUALQUIER manzana (roja, verde, amarilla)
    # Esto es amplio para asegurarnos de que encuentre la manzana
    limite_bajo = np.array([0, 50, 50])
    limite_alto = np.array([180, 255, 255])
    mascara = cv2.inRange(hsv_frame, limite_bajo, limite_alto)
    
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        contorno_manzana = max(contornos, key=cv2.contourArea)
        
        if cv2.contourArea(contorno_manzana) > 2000:
            # Calculamos el color promedio dentro del contorno de la manzana
            mascara_manzana = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(mascara_manzana, [contorno_manzana], -1, 255, -1)
            color_promedio_hsv = cv2.mean(hsv_frame, mask=mascara_manzana)
            hue = color_promedio_hsv[0]

            # Imprimimos el valor HUE para poder calibrarlo
            print(f"Hue detectado: {hue:.2f}")

           # --- LÓGICA DE CLASIFICACIÓN FINAL PARA COLOR ---
            clasificacion_color = ""
            if 0 <= hue <= 15: # Rango para Rojo
                clasificacion_color = "Roja"
            elif 38 <= hue <= 50: # Rango para Verde
                clasificacion_color = "Verde"
            elif 20 <= hue <= 35: # Rango para Amarillo
                clasificacion_color = "Amarilla"
            else:
                clasificacion_color = "No definido"
            
            # Mostramos un recuadro y el texto en la imagen
            x, y, w, h = cv2.boundingRect(contorno_manzana)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, clasificacion_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow('Calibracion de Color', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()