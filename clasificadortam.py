import cv2
import numpy as np


DIAMETRO_MONEDA_MM = 25.5
MANZANA_PEQUENA_MM = 50
MANZANA_MEDIANA_MM = 61.5

cap = cv2.VideoCapture(0)
print("Coloca la moneda de 1 sol peruano sobre fondo claro. Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Calibración automática con la moneda ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Mejor contraste para fondo claro
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (11, 11), 2)
    # Invertimos para que la moneda oscura sobre fondo claro se detecte mejor
    gray_inv = cv2.bitwise_not(gray)

    # Prueba ambos: original e invertido
    circles = cv2.HoughCircles(
        gray_inv, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=60, param2=25, minRadius=12, maxRadius=60
    )

    PIXELES_POR_MM = None
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Selecciona el círculo más grande (probable moneda)
        max_r = 0
        moneda = None
        for (x, y, r) in circles:
            if r > max_r:
                max_r = r
                moneda = (x, y, r)
        if moneda:
            x, y, r = moneda
            diametro_px = r * 2
            PIXELES_POR_MM = diametro_px / DIAMETRO_MONEDA_MM
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)
            cv2.putText(frame, "Moneda detectada", (x - 40, y - r - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(frame, f"Escala: {PIXELES_POR_MM:.2f} px/mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # --- Clasificación de la manzana ---
    if PIXELES_POR_MM:
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        limite_bajo = np.array([0, 50, 50])
        limite_alto = np.array([180, 255, 255])
        mascara = cv2.inRange(hsv_frame, limite_bajo, limite_alto)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, np.ones((5,5), np.uint8))
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contornos:
            contorno_manzana = max(contornos, key=cv2.contourArea)
            if cv2.contourArea(contorno_manzana) > 2000:
                x, y, w, h = cv2.boundingRect(contorno_manzana)
                ancho_real_mm = w / PIXELES_POR_MM

                clasificacion_tamano = ""
                if ancho_real_mm < MANZANA_PEQUENA_MM:
                    clasificacion_tamano = "Pequena"
                elif ancho_real_mm < MANZANA_MEDIANA_MM:
                    clasificacion_tamano = "Mediana"
                else:
                    clasificacion_tamano = "Grande"

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                texto = f"Tamano: {clasificacion_tamano} ({ancho_real_mm:.1f} mm)"
                cv2.putText(frame, texto, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Clasificador', frame)
    cv2.imshow('Gris invertido', gray_inv)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()