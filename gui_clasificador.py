import sys
import cv2
import numpy as np
import tensorflow as tf
import serial
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from collections import OrderedDict, deque
from scipy.spatial import distance as dist

# --- CLASE Rastreador de Objetos (para IDs estables) ---
class CentroidTracker:
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        if objectID in self.objects: del self.objects[objectID]
        if objectID in self.disappeared: del self.disappeared[objectID]
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            return self.objects
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0); cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
        if len(self.objects) == 0:
            for i in range(len(inputCentroids)): self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys()); objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
            usedRows, usedCols = set(), set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            else:
                for col in unusedCols: self.register(inputCentroids[col])
        return self.objects

# --- CONFIGURACIÓN GENERAL ---
PIXELES_POR_MM = 3.2 
MANZANA_PEQUENA_MM = 65
MANZANA_MEDIANA_MM = 80
PUERTO_ARDUINO = 'COM9'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3

# --- CARGA DE MODELOS ---
model_loaded = False
try:
    classification_model = tf.keras.models.load_model('modelo_manzanas.h5')
    CLASS_NAMES = ['intermedia', 'madura', 'verde']
    IMG_HEIGHT, IMG_WIDTH = 224, 224
    model_loaded = True
    print("Modelo de clasificación cargado.")
except IOError:
    print("ADVERTENCIA: No se pudo cargar 'modelo_manzanas.h5'.")

try:
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("Modelo YOLOv3 cargado correctamente.")
except Exception as e:
    print(f"Error fatal al cargar YOLOv3: {e}")
    sys.exit()

class ClasificadorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = CentroidTracker()
        self.objetos_posicion_previa = {}
        self.medidas_tamano = {}
        self.arduino = None
        self.initUI()
        self.initCamera()
        self.initArduino()

    def initUI(self):
        self.setWindowTitle("Clasificador de Manzanas v2.2 (Final)")
        self.setGeometry(100, 100, 1200, 700)
        self.setStyleSheet("QWidget { background-color: #2E2E2E; color: #E0E0E0; } QLabel { color: #E0E0E0; } QPushButton { background-color: #4A4A4A; border: 1px solid #6A6A6A; padding: 8px; border-radius: 4px; } QPushButton:hover { background-color: #5A5A5A; } QFrame { background-color: #4A4A4A; }")
        
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        self.video_label = QLabel("Presiona 'Iniciar'"); self.video_label.setAlignment(Qt.AlignCenter); self.video_label.setFont(QFont("Arial", 14)); main_layout.addWidget(self.video_label, 7)
        control_panel_layout = QVBoxLayout(); main_layout.addLayout(control_panel_layout, 3)
        
        title_label = QLabel("Panel de Control"); title_label.setFont(QFont("Arial", 18, QFont.Bold)); title_label.setAlignment(Qt.AlignCenter); control_panel_layout.addWidget(title_label)
        
        self.results_display = QLabel("Esperando inicio..."); self.results_display.setFont(QFont("Arial", 12)); self.results_display.setAlignment(Qt.AlignTop | Qt.AlignLeft); control_panel_layout.addWidget(self.results_display); control_panel_layout.addStretch()
        
        button_layout = QHBoxLayout(); self.toggle_button = QPushButton("Iniciar", self); self.toggle_button.clicked.connect(self.toggle_camera); self.quit_button = QPushButton("Salir", self); self.quit_button.clicked.connect(self.close); button_layout.addWidget(self.toggle_button); button_layout.addWidget(self.quit_button); control_panel_layout.addLayout(button_layout)
        
    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        
    def initArduino(self):
        try:
            self.arduino = serial.Serial(PUERTO_ARDUINO, 9600, timeout=1)
            time.sleep(2)
            print(f"Conexión con Arduino en {PUERTO_ARDUINO} establecida.")
        except Exception as e:
            print(f"ADVERTENCIA: No se pudo conectar con Arduino. {e}")
            self.arduino = None

    def toggle_camera(self):
        if not self.is_running:
            if not self.cap.isOpened(): self.cap.open(0)
            self.timer.start(30); self.toggle_button.setText("Detener"); self.is_running = True
        else:
            self.timer.stop(); self.toggle_button.setText("Iniciar"); self.is_running = False
            self.video_label.setText("Cámara detenida.")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        output_frame = frame.copy()
        h, w, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        linea_y = h * 2 // 3
        cv2.line(output_frame, (0, linea_y), (w, linea_y), (0, 0, 255), 2)

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
                    center_x, center_y = int(detection[0] * w), int(detection[1] * h)
                    w_box, h_box = int(detection[2] * w), int(detection[3] * h)
                    x, y = int(center_x - w_box / 2), int(center_y - h_box / 2)
                    boxes.append([x, y, w_box, h_box])
                    confidences.append(float(confidence))

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        
        rects = [boxes[i] for i in indexes.flatten()] if len(indexes) > 0 else []
        objects = self.tracker.update([ (x, y, x + w_box, y + h_box) for (x, y, w_box, h_box) in rects ])
        
        all_results_text = []
        for (objectID, centroid) in objects.items():
            for rect in rects:
                x, y, w_box, h_box = rect
                if centroid[0] > x and centroid[0] < x + w_box and centroid[1] > y and centroid[1] < y + h_box:
                    x, y = max(0, x), max(0, y)
                    
                    ancho_real_mm = w_box / PIXELES_POR_MM
                    if objectID not in self.medidas_tamano:
                        self.medidas_tamano[objectID] = deque(maxlen=10)
                    self.medidas_tamano[objectID].append(ancho_real_mm)
                    ancho_promedio_mm = np.mean(self.medidas_tamano[objectID])
                    
                    clasificacion_tamano = "Grande"
                    if ancho_promedio_mm < MANZANA_PEQUENA_MM: clasificacion_tamano = "Pequena"
                    elif ancho_promedio_mm < MANZANA_MEDIANA_MM: clasificacion_tamano = "Mediana"
                    
                    mask_forma = np.zeros(hsv_frame.shape[:2], dtype="uint8")
                    cv2.rectangle(mask_forma, (x, y), (x + w_box, y + h_box), 255, -1)
                    mask_color_valido = cv2.inRange(hsv_frame, np.array([0, 70, 50]), np.array([180, 255, 255]))
                    mask_final = cv2.bitwise_and(mask_forma, mask_color_valido)
                    
                    hue_promedio = -1
                    if np.any(mask_final):
                        hue_promedio = cv2.mean(hsv_frame, mask=mask_final)[0]
                    
                    clasificacion_color = "No definido"
                    if 0 <= hue_promedio <= 18 or 170 <= hue_promedio <= 180: clasificacion_color = "Roja"
                    elif 35 <= hue_promedio <= 75: clasificacion_color = "Verde"
                    elif 19 <= hue_promedio <= 34: clasificacion_color = "Amarilla"
                    
                    clasificacion_madurez = "N/A"; confianza = 0
                    if model_loaded:
                        roi = frame[y:y+h_box, x:x+w_box]
                        if roi.size > 0:
                            img_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
                            img_array = tf.keras.utils.img_to_array(img_resized); img_batch = np.expand_dims(img_array, axis=0)
                            prediction = classification_model.predict(img_batch, verbose=0); score = tf.nn.softmax(prediction[0])
                            clasificacion_madurez = CLASS_NAMES[np.argmax(score)]; confianza = 100 * np.max(score)

                    resultado_manzana = (f"<b>Manzana {objectID}</b><br>"
                                         f"&nbsp;&nbsp;Tamaño: {clasificacion_tamano} ({ancho_promedio_mm:.1f} mm)<br>"
                                         f"&nbsp;&nbsp;Color: {clasificacion_color}<br>"
                                         f"&nbsp;&nbsp;Madurez: {clasificacion_madurez} ({confianza:.1f}%)")
                    all_results_text.append(resultado_manzana)

                    cv2.rectangle(output_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"ID {objectID}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    posicion_previa = self.objetos_posicion_previa.get(objectID, 0)
                    if posicion_previa < linea_y and centroid[1] >= linea_y:
                        if self.arduino:
                            comando = clasificacion_madurez[0].upper()
                            self.arduino.write(comando.encode())
                            print(f"--> ¡CRUCE! Manzana {objectID} ({clasificacion_madurez}). Orden enviada: {comando}")
                    break
        
        self.objetos_posicion_previa = {obj_id: center[1] for obj_id, center in objects.items()}

        if all_results_text: self.results_display.setText("<br><br>".join(all_results_text))
        else: self.results_display.setText("No se detectan manzanas")
        
        rgb_image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape; bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        if self.arduino: self.arduino.close()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = ClasificadorApp()
    main_window.show()
    sys.exit(app.exec_())

