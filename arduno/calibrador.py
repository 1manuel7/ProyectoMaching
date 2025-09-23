import cv2
import numpy as np

# --- CONFIGURACIÓN ---
DIAMETRO_MONEDA_MM = 25.5

cap = cv2.VideoCapture(0)
print("Coloca la moneda de 1 sol peruano en la imagen. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertimos a escala de grises y aplicamos desenfoque
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 2)

    # Detecta círculos usando HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=10, maxRadius=100)

    PIXELES_POR_MM = None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Suponemos que el círculo más grande es la moneda
            if r > 10:  # Ajusta este valor si es necesario
                diametro_px = r * 2
                PIXELES_POR_MM = diametro_px / DIAMETRO_MONEDA_MM
                cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
                cv2.putText(frame, "Moneda detectada", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                cv2.putText(frame, f"Escala: {PIXELES_POR_MM:.2f} px/mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                break

    cv2.imshow('Calibracion con moneda', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()