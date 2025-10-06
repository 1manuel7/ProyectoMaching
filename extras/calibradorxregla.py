import cv2
import numpy as np

# --- CONFIGURACIÓN ---
# Pon aquí el ancho real de tu cuadrado negro en milímetros
ANCHO_MARCADOR_MM = 20.0

cap = cv2.VideoCapture(0)

print("--- Calibracion con Marcador ---")
print("Coloca el cuadrado negro en el centro y anota el valor.")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mascara = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contornos:
        contorno_marcador = max(contornos, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contorno_marcador)
        
        if w > 20: 
            pixels_por_mm = w / ANCHO_MARCADOR_MM
            
            print(f"VALOR DE CALIBRACION (pixeles_por_mm): {pixels_por_mm:.4f}")
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Valor: {pixels_por_mm:.4f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Calibrador con Marcador", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()