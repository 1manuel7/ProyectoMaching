import cv2
import numpy as np
import tensorflow as tf
import serial
import time

# --- CONFIGURACIÓN ---
PIXELES_POR_MM = 1.8500 # Usa el valor de tu calibración final
MANZANA_PEQUENA_MM = 65
MANZANA_MEDIANA_MM = 80
PUERTO_ARDUINO = 'COM9' # <--- CAMBIA ESTO por tu puerto correcto

# --- CONEXIÓN CON ARDUINO ---
try:
    arduino = serial.Serial(PUERTO_ARDUINO, 9600, timeout=1)
    time.sleep(2)
    print(f"Conexión con Arduino en {PUERTO_ARDUINO} establecida.")
except Exception as e:
    print(f"ADVERTENCIA: No se pudo conectar con Arduino. {e}")
    arduino = None

# --- CARGAR MODELO DE IA ---
print("Cargando modelo de IA...")
model = tf.keras.models.load_model('modelo_manzanas.h5')
CLASS_NAMES = ['intermedia', 'madura', 'verde']
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- INICIO DEL PROGRAMA ---
cap = cv2.VideoCapture(0)
print("Sistema de clasificacion final iniciado. Presiona 'q' para salir.")

# Diccionario para rastrear la posición previa de los objetos y evitar dobles envíos
objetos_posicion_previa = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    output_frame = frame.copy()
    
    # --- AÑADIR ESTA LÍNEA PARA EL TÍTULO ---
    cv2.putText(output_frame, "Clasificador de Manzanas", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    h, w, _ = output_frame.shape

    linea_y = h * 2 // 3
    cv2.line(output_frame, (0, linea_y), (w, linea_y), (0, 0, 255), 2)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    limite_bajo = np.array([0, 50, 50])
    limite_alto = np.array([180, 255, 255])
    mascara = cv2.inRange(hsv_frame, limite_bajo, limite_alto)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    objetos_detectados_actualmente = {}

    for i, contorno in enumerate(contornos):
        area = cv2.contourArea(contorno)
        if area > 5000:
            x, y, w_bbox, h_bbox = cv2.boundingRect(contorno)
            centro_y_manzana = y + h_bbox // 2
            objetos_detectados_actualmente[i] = centro_y_manzana

            # Clasificación por Tamaño
            ancho_real_mm = w_bbox / PIXELES_POR_MM
            clasificacion_tamano = "Grande"
            if ancho_real_mm < MANZANA_PEQUENA_MM: clasificacion_tamano = "Pequena"
            elif ancho_real_mm < MANZANA_MEDIANA_MM: clasificacion_tamano = "Mediana"
            
            # Clasificación por Color
            mascara_color = np.zeros(frame.shape[:2], dtype="uint8")
            cv2.drawContours(mascara_color, [contorno], -1, 255, -1)
            hue = cv2.mean(hsv_frame, mask=mascara_color)[0]
            clasificacion_color = "No definido"
            if 0 <= hue <= 15: clasificacion_color = "Roja"
            elif 38 <= hue <= 50: clasificacion_color = "Verde"
            elif 20 <= hue <= 35: clasificacion_color = "Amarilla"
            
            # Clasificación por Madurez (IA)
            roi = frame[y:y+h_bbox, x:x+w_bbox]
            img_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
            img_array = tf.keras.utils.img_to_array(img_resized)
            img_batch = np.expand_dims(img_array, axis=0)
            prediction = model.predict(img_batch, verbose=0)
            score = tf.nn.softmax(prediction[0])
            clasificacion_madurez = CLASS_NAMES[np.argmax(score)]
            confianza = 100 * np.max(score)

            # Dibujar la información
            cv2.rectangle(output_frame, (x, y), (x + w_bbox, y + h_bbox), (0, 255, 0), 2)
            texto1 = f"T: {clasificacion_tamano} ({ancho_real_mm:.1f} mm)"
            texto2 = f"C: {clasificacion_color}, M: {clasificacion_madurez} ({confianza:.1f}%)"
            cv2.putText(output_frame, texto1, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(output_frame, texto2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Acción del Servo
            posicion_previa = objetos_posicion_previa.get(i, 0)
            if posicion_previa < linea_y and centro_y_manzana >= linea_y:
                if arduino:
                    comando = clasificacion_madurez[0].upper()
                    arduino.write(comando.encode())
                    print(f"--> ¡CRUCE! Manzana {i} clasificada como {clasificacion_madurez}. Orden enviada: {comando}")

    objetos_posicion_previa = objetos_detectados_actualmente.copy()

    cv2.imshow('Clasificador Final IA', output_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if arduino:
    arduino.close()
cap.release()
cv2.destroyAllWindows()