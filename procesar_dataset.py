import cv2
import numpy as np
import os

# --- CONFIGURACIÓN ---
CARPETA_ENTRADA = 'fotos_sin_procesar'
CARPETA_SALIDA = 'dataset'
CATEGORIAS = ["madura", "intermedia", "verde"]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# --- CARGA DEL MODELO YOLOv3 ---
print("Cargando modelo de detección YOLOv3...")
try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("Modelo YOLOv3 cargado correctamente.")
except Exception as e:
    print(f"Error fatal al cargar YOLOv3: {e}")
    print("Asegúrate de tener 'yolov3.weights', 'yolov3.cfg' y 'coco.names' en la misma carpeta.")
    exit()

# --- INICIO DEL PROCESAMIENTO ---
print("Iniciando procesamiento automático del dataset con YOLO...")

for categoria in CATEGORIAS:
    path_entrada = os.path.join(CARPETA_ENTRADA, categoria)
    path_salida = os.path.join(CARPETA_SALIDA, categoria)

    if not os.path.exists(path_entrada):
        print(f"ADVERTENCIA: La carpeta de entrada no existe: {path_entrada}")
        continue
        
    if not os.path.exists(path_salida):
        os.makedirs(path_salida)

    print(f"\nProcesando categoría: {categoria}")

    for nombre_archivo in os.listdir(path_entrada):
        ruta_archivo = os.path.join(path_entrada, nombre_archivo)
        
        frame = cv2.imread(ruta_archivo)
        if frame is None:
            print(f"  - Error al leer {nombre_archivo}, omitiendo.")
            continue

        h, w, _ = frame.shape
        
        # Detección con YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        boxes, confidences = [], []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if classes[class_id] == "apple" and confidence > CONFIDENCE_THRESHOLD:
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    w_box = int(detection[2] * w)
                    h_box = int(detection[3] * h)
                    x = int(center_x - w_box / 2)
                    y = int(center_y - h_box / 2)
                    boxes.append([x, y, w_box, h_box])
                    confidences.append(float(confidence))

        # Supresión de no máximos para eliminar cajas duplicadas
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w_box, h_box = boxes[i]
                
                # Asegurarse de que las coordenadas no se salgan de la imagen
                x, y = max(0, x), max(0, y)
                
                # Recortar la manzana (Region of Interest - ROI)
                roi = frame[y:y+h_box, x:x+w_box]

                if roi.size == 0:
                    continue
                
                # Guardar la imagen recortada con un nombre único
                nombre_base, extension = os.path.splitext(nombre_archivo)
                ruta_guardado = os.path.join(path_salida, f"{nombre_base}_recorte_{i}{extension}")
                cv2.imwrite(ruta_guardado, roi)
                print(f"  + Manzana recortada y guardada: {ruta_guardado}")
        else:
            print(f"  - No se detectó ninguna manzana en {nombre_archivo}")

print("\n¡Procesamiento completado! Tu dataset está listo en la carpeta 'dataset'.")