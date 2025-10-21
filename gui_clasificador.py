import sys
import cv2
import numpy as np
import tensorflow as tf
import serial
import time
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from collections import OrderedDict, deque
from scipy.spatial import distance as dist

# --- (La clase CentroidTracker y la carga de modelos no cambian) ---
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

PIXELES_POR_MM = 1.6500
MANZANA_PEQUENA_MM = 113
MANZANA_MEDIANA_MM = 118
PUERTO_ARDUINO = 'COM9'
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.3
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
    with open("coco.names", "r") as f: classes = [line.strip() for line in f.readlines()]
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
        self.predicciones_recientes = {}
        self.arduino = None
        
        # --- Banderas para los checkboxes ---
        self.show_tamano = True
        self.show_color = True
        self.show_madurez = True
        
        self.initUI()
        self.initCamera()
        self.initArduino()

    def initUI(self):
        self.setWindowTitle("Clasificador de Manzanas v2.3 (Estable)")
        self.setGeometry(100, 100, 1200, 700)

        # --- HOJA DE ESTILOS CSS PARA LA APLICACIÓN ---
        style_sheet = """
            QMainWindow {
                background-color: #2E2E2E; /* Fondo principal oscuro */
            }
            #PanelControl {
                background-color: #383838;
                border-radius: 8px;
            }
            QLabel {
                color: #E0E0E0;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            #TituloPanel {
                font-size: 18px;
                font-weight: bold;
                color: #FFFFFF;
                padding: 10px;
                border-bottom: 1px solid #4A4A4A;
            }
            #VideoLabel {
                background-color: #000000; /* Fondo negro para el video */
            }
            #ResultadosDisplay {
                padding: 10px;
                font-size: 12px;
            }
            QPushButton {
                background-color: #4A4A4A;
                color: #FFFFFF;
                font-weight: bold;
                border: 1px solid #6A6A6A;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #5A5A5A;
            }
            QPushButton:pressed {
                background-color: #6A6A6A; /* Más oscuro al presionar */
            }
            QCheckBox {
                color: #C0C0C0;
                font-size: 11px;
            }
        """
        self.setStyleSheet(style_sheet)
        
        # --- ESTRUCTURA DE LAYOUTS ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Panel de Video (Izquierda)
        self.video_label = QLabel("Presiona 'Iniciar'");
        self.video_label.setAlignment(Qt.AlignCenter);
        self.video_label.setFont(QFont("Arial", 14));
        self.video_label.setObjectName("VideoLabel") # ID para CSS
        main_layout.addWidget(self.video_label, 7) # 70% del espacio

        # Panel de Control (Derecha)
        panel_control_widget = QWidget()
        panel_control_widget.setObjectName("PanelControl") # ID para CSS
        control_panel_layout = QVBoxLayout(panel_control_widget)
        main_layout.addWidget(panel_control_widget, 3) # 30% del espacio
        
        title_label = QLabel("Panel de Control");
        title_label.setObjectName("TituloPanel") # ID para CSS
        title_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(title_label)
        
        # --- Checkboxes de Filtro ---
        checkbox_layout = QHBoxLayout()
        checkbox_layout.setContentsMargins(10, 0, 10, 10)
        checkbox_layout.addWidget(QLabel("Mostrar:"))
        self.check_tamano = QCheckBox("Tamaño"); self.check_tamano.setChecked(True)
        self.check_color = QCheckBox("Color"); self.check_color.setChecked(True)
        self.check_madurez = QCheckBox("Madurez"); self.check_madurez.setChecked(True)
        
        self.check_tamano.stateChanged.connect(self.update_display_flags)
        self.check_color.stateChanged.connect(self.update_display_flags)
        self.check_madurez.stateChanged.connect(self.update_display_flags)
        
        checkbox_layout.addWidget(self.check_tamano)
        checkbox_layout.addWidget(self.check_color)
        checkbox_layout.addWidget(self.check_madurez)
        control_panel_layout.addLayout(checkbox_layout)
        
        # --- Panel de Resultados Dinámico ---
        self.results_display = QLabel("Esperando inicio...")
        self.results_display.setObjectName("ResultadosDisplay") # ID para CSS
        self.results_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_display.setWordWrap(True)
        control_panel_layout.addWidget(self.results_display)
        
        control_panel_layout.addStretch()
        
        # Botones
        button_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Iniciar"); self.toggle_button.clicked.connect(self.toggle_camera)
        self.quit_button = QPushButton("Salir"); self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.toggle_button)
        button_layout.addWidget(self.quit_button)
        control_panel_layout.addLayout(button_layout)
        
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

    def update_display_flags(self):
        self.show_tamano = self.check_tamano.isChecked()
        self.show_color = self.check_color.isChecked()
        self.show_madurez = self.check_madurez.isChecked()
            
    def toggle_camera(self):
        if not self.is_running:
            if not self.cap.isOpened(): self.cap.open(0)
            self.timer.start(30); self.toggle_button.setText("Detener"); self.is_running = True
        else:
            self.timer.stop(); self.toggle_button.setText("Iniciar"); self.is_running = False
            self.video_label.setText("Cámara detenida.")
            self.results_display.setText("Detección detenida.")
            # Limpiar rastreadores al detener
            self.tracker = CentroidTracker()
            self.objetos_posicion_previa.clear()
            self.medidas_tamano.clear()
            self.predicciones_recientes.clear()

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
                    
                    # --- Medición Precisa de Tamaño ---
                    roi_hsv_mask = hsv_frame[y:y+h_box, x:x+w_box]
                    mask_roi = cv2.inRange(roi_hsv_mask, np.array([0, 40, 40]), np.array([180, 255, 255]))
                    contornos_roi, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contornos_roi: continue
                    contorno_manzana = max(contornos_roi, key=cv2.contourArea)
                    ((cx, cy), radio) = cv2.minEnclosingCircle(contorno_manzana)
                    diametro_real_mm = (radio * 2) / PIXELES_POR_MM
                    if objectID not in self.medidas_tamano: self.medidas_tamano[objectID] = deque(maxlen=10)
                    self.medidas_tamano[objectID].append(diametro_real_mm)
                    diametro_promedio_mm = np.mean(self.medidas_tamano[objectID])
                    
                    clasificacion_tamano = "Grande"
                    if diametro_promedio_mm < MANZANA_PEQUENA_MM: clasificacion_tamano = "Pequena"
                    elif diametro_promedio_mm < MANZANA_MEDIANA_MM: clasificacion_tamano = "Mediana"
                    
                    # --- Medición de Color Robusta ---
                    mask_final = np.zeros(hsv_frame.shape[:2], dtype="uint8")
                    contorno_manzana[:, :, 0] += x
                    contorno_manzana[:, :, 1] += y
                    cv2.drawContours(mask_final, [contorno_manzana], -1, 255, -1)
                    mask_color_valido = cv2.inRange(hsv_frame, np.array([0, 70, 50]), np.array([180, 255, 255]))
                    mask_final = cv2.bitwise_and(mask_final, mask_color_valido)
                    hue_promedio = -1
                    if np.any(mask_final): hue_promedio = cv2.mean(hsv_frame, mask=mask_final)[0]
                    
                    clasificacion_color = "No definido"
                    if 0 <= hue_promedio <= 18 or 170 <= hue_promedio <= 180: clasificacion_color = "Roja"
                    elif 35 <= hue_promedio <= 75: clasificacion_color = "Verde"
                    elif 19 <= hue_promedio <= 34: clasificacion_color = "Amarilla"
                    
                    # --- Clasificación de Madurez Estable ---
                    clasificacion_madurez, confianza = "Analizando...", 0
                    if model_loaded:
                        roi = frame[y:y+h_box, x:x+w_box]
                        if roi.size > 0:
                            img_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
                            img_array = tf.keras.utils.img_to_array(img_resized); img_batch = np.expand_dims(img_array, axis=0)
                            prediction = classification_model.predict(img_batch, verbose=0); score = tf.nn.softmax(prediction[0])
                            pred_actual = CLASS_NAMES[np.argmax(score)]
                            if objectID not in self.predicciones_recientes: self.predicciones_recientes[objectID] = deque(maxlen=5)
                            self.predicciones_recientes[objectID].append(pred_actual)
                            if len(self.predicciones_recientes[objectID]) == 5:
                                clasificacion_madurez = max(set(self.predicciones_recientes[objectID]), key=list(self.predicciones_recientes[objectID]).count)
                                confianza = 100 * np.max(score)

                    # --- NUEVO: Construcción de la "Tarjeta" de Resultados con HTML ---
                    resultado_partes = [
                        f"<div style='background-color:#4A4A4A; border-radius: 8px; padding: 10px; margin-bottom: 10px;'>"
                        f"<b style='color: #FFFFFF; font-size: 14px;'>Manzana {objectID}</b><hr style='border-color: #555555;'>"
                    ]
                    
                    if self.show_tamano:
                        resultado_partes.append(f"Tamaño: {clasificacion_tamano} ({diametro_promedio_mm:.1f} mm)")
                    
                    if self.show_color:
                        color_hex = "#E0E0E0" # Gris claro por defecto
                        if clasificacion_color == "Roja": color_hex = "#F87171"
                        elif clasificacion_color == "Verde": color_hex = "#4ADE80"
                        elif clasificacion_color == "Amarilla": color_hex = "#FACC15"
                        resultado_partes.append(f"Color: <b style='color: {color_hex};'>{clasificacion_color}</b>")
                    
                    if self.show_madurez and model_loaded:
                        conf_color = "#F87171" # Rojo (baja confianza)
                        if confianza > 75: conf_color = "#4ADE80" # Verde (alta confianza)
                        elif confianza > 40: conf_color = "#FACC15" # Amarillo (media confianza)
                        resultado_partes.append(f"Madurez: {clasificacion_madurez} (<span style='color: {conf_color};'>{confianza:.1f}%</span>)")

                    resultado_partes.append("</div>")
                    all_results_text.append("".join(resultado_partes))
                    
                    # Dibujar en el video
                    cv2.rectangle(output_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"ID {objectID}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
        
        # Actualizar el panel de resultados
        if all_results_text:
            self.results_display.setText("".join(all_results_text))
        else:
            self.results_display.setText("No se detectan manzanas")
        
        # Convertir y mostrar el frame
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