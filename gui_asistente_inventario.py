import sys
import cv2
import numpy as np
import tensorflow as tf
import serial
import time
import csv
import os
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame, QLineEdit
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from collections import OrderedDict, deque, Counter
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

# --- CONFIGURACIÓN GENERAL ---
PIXELES_POR_MM = 3.2 
MANZANA_PEQUENA_MM = 65
MANZANA_MEDIANA_MM = 80
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
    with open("coco.names", "r") as f: classes = [line.strip() for line in f.readlines()]
    print("Modelo YOLOv3 cargado correctamente.")
except Exception as e:
    print(f"Error fatal al cargar YOLOv3: {e}")
    sys.exit()

class AsistenteInventarioApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = CentroidTracker()
        self.medidas_tamano = {}
        self.predicciones_recientes = {}
        
        # --- Variables de estado para el inventario ---
        self.lote_actual_conteo = Counter()
        self.lote_actual_valor = 0.0
        self.total_dia_conteo = Counter()
        self.total_dia_valor = 0.0
        
        self.initUI()
        self.initCamera()
        
    def initUI(self):
        self.setWindowTitle("Asistente de Inventario de Manzanas v3.0")
        self.setGeometry(100, 100, 1200, 750) 

        # --- Hoja de Estilos ---
        style_sheet = """
            QWidget { background-color: #2E2E2E; color: #E0E0E0; }
            #PanelControl { background-color: #383838; border-radius: 8px; }
            QLabel { color: #E0E0E0; font-family: 'Segoe UI', Arial; }
            #TituloPanel, #TituloPrecios, #TituloLote, #TituloDia {
                font-size: 16px; font-weight: bold; color: #FFFFFF; padding: 10px;
                border-bottom: 1px solid #4A4A4A;
            }
            #VideoLabel { background-color: #000000; }
            #ResultadosDisplay, #TotalDisplay { padding: 10px; font-size: 12px; }
            QPushButton { 
                background-color: #4A4A4A; color: #FFFFFF; font-weight: bold; 
                border: 1px solid #6A6A6A; padding: 10px; border-radius: 4px;
            }
            QPushButton:hover { background-color: #5A5A5A; }
            QPushButton:pressed { background-color: #6A6A6A; }
            QLineEdit { 
                background-color: #5A5A5A; color: #FFFFFF; border: 1px solid #6A6A6A; 
                border-radius: 4px; padding: 5px; font-weight: bold;
            }
        """
        self.setStyleSheet(style_sheet)
        
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget); main_layout.setSpacing(15); main_layout.setContentsMargins(15, 15, 15, 15)
        
        # --- Panel de Video (Izquierda) ---
        self.video_label = QLabel("Presiona 'Iniciar'"); self.video_label.setAlignment(Qt.AlignCenter); self.video_label.setFont(QFont("Arial", 14)); self.video_label.setObjectName("VideoLabel"); main_layout.addWidget(self.video_label, 7)
        
        # --- Panel de Control (Derecha) ---
        panel_control_widget = QWidget(); panel_control_widget.setObjectName("PanelControl"); control_panel_layout = QVBoxLayout(panel_control_widget); main_layout.addWidget(panel_control_widget, 3)
        
        # --- CORRECCIÓN DE LOS TÍTULOS ---
        title_label = QLabel("Panel de Control")
        title_label.setObjectName("TituloPanel")
        title_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(title_label)
        
        precios_label = QLabel("Configuración de Precios (S/.)")
        precios_label.setObjectName("TituloPrecios")
        precios_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(precios_label)

        precio_layout = QHBoxLayout(); precio_layout.setContentsMargins(10, 5, 10, 10)
        precio_layout.addWidget(QLabel("Madura:")); self.precio_madura = QLineEdit("1.50"); self.precio_madura.setFixedWidth(60); precio_layout.addWidget(self.precio_madura)
        precio_layout.addWidget(QLabel("Intermedia:")); self.precio_intermedia = QLineEdit("1.00"); self.precio_intermedia.setFixedWidth(60); precio_layout.addWidget(self.precio_intermedia)
        precio_layout.addWidget(QLabel("Verde:")); self.precio_verde = QLineEdit("0.50"); self.precio_verde.setFixedWidth(60); precio_layout.addWidget(self.precio_verde)
        control_panel_layout.addLayout(precio_layout)

        lote_label = QLabel("Lote Actual (En Cámara)")
        lote_label.setObjectName("TituloLote")
        lote_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(lote_label)
        
        self.lote_display = QLabel("No se detectan manzanas"); self.lote_display.setObjectName("ResultadosDisplay"); self.lote_display.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.lote_display.setWordWrap(True)
        control_panel_layout.addWidget(self.lote_display)
        
        self.add_to_total_button = QPushButton("Añadir Lote al Total del Día"); self.add_to_total_button.clicked.connect(self.anadir_al_total)
        control_panel_layout.addWidget(self.add_to_total_button)
        
        control_panel_layout.addWidget(QFrame(frameShape=QFrame.HLine))

        total_label = QLabel("Total del Día")
        total_label.setObjectName("TituloDia")
        total_label.setAlignment(Qt.AlignCenter)
        control_panel_layout.addWidget(total_label)
        
        self.total_display = QLabel("Total: 0 manzanas (S/ 0.00)"); self.total_display.setObjectName("TotalDisplay"); self.total_display.setAlignment(Qt.AlignTop | Qt.AlignLeft); self.total_display.setWordWrap(True)
        control_panel_layout.addWidget(self.total_display)
        
        self.export_button = QPushButton("Exportar y Reiniciar Día"); self.export_button.clicked.connect(self.exportar_y_reiniciar)
        control_panel_layout.addWidget(self.export_button)
        
        control_panel_layout.addStretch()
        
        button_layout = QHBoxLayout(); self.toggle_button = QPushButton("Iniciar"); self.toggle_button.clicked.connect(self.toggle_camera); self.quit_button = QPushButton("Salir"); self.quit_button.clicked.connect(self.close); button_layout.addWidget(self.toggle_button); button_layout.addWidget(self.quit_button); control_panel_layout.addLayout(button_layout)
        
    def initCamera(self):
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.is_running = False
        
    # --- initArduino() ya no es necesario ---

    def toggle_camera(self):
        if not self.is_running:
            if not self.cap.isOpened(): self.cap.open(0)
            self.timer.start(30); self.toggle_button.setText("Detener"); self.is_running = True
        else:
            self.timer.stop(); self.toggle_button.setText("Iniciar"); self.is_running = False
            self.video_label.setText("Cámara detenida.")
            self.lote_display.setText("Detección detenida.")
            # Limpiar rastreadores al detener
            self.tracker = CentroidTracker()
            self.medidas_tamano.clear()
            self.predicciones_recientes.clear()

    def get_precios(self):
        try: p_madura = float(self.precio_madura.text())
        except ValueError: p_madura = 0.0
        try: p_intermedia = float(self.precio_intermedia.text())
        except ValueError: p_intermedia = 0.0
        try: p_verde = float(self.precio_verde.text())
        except ValueError: p_verde = 0.0
        return p_madura, p_intermedia, p_verde

    def anadir_al_total(self):
        if not self.lote_actual_conteo:
            print("No hay manzanas en el lote actual para añadir.")
            return

        self.total_dia_conteo.update(self.lote_actual_conteo)
        self.total_dia_valor += self.lote_actual_valor
        
        print(f"Lote añadido al total. Total del día: {self.total_dia_conteo}")
        self.actualizar_display_total()
        
        self.tracker = CentroidTracker()
        self.lote_display.setText("Lote añadido. Listo para el siguiente.")
        self.medidas_tamano.clear()
        self.predicciones_recientes.clear()

    def actualizar_display_total(self):
        # --- CORRECCIÓN: Calcular el total sumando las categorías de madurez ---
        total_manzanas = self.total_dia_conteo.get('madura', 0) + self.total_dia_conteo.get('intermedia', 0) + self.total_dia_conteo.get('verde', 0)
        
        reporte_html = f"<b>Total Acumulado: {total_manzanas} manzanas</b><br>"
        reporte_html += f"<b>Valor Total: S/ {self.total_dia_valor:.2f}</b><br><br>"
        reporte_html += "<b>Detalle por Madurez:</b><br>"
        reporte_html += f"&nbsp;&nbsp; Maduras: {self.total_dia_conteo.get('madura', 0)}<br>" # <-- CORRECCIÓN: Arreglado typo
        reporte_html += f"&nbsp;&nbsp; Intermedias: {self.total_dia_conteo.get('intermedia', 0)}<br>"
        reporte_html += f"&nbsp;&nbsp; Verdes: {self.total_dia_conteo.get('verde', 0)}<br><br>"
        reporte_html += "<b>Detalle por Tamaño:</b><br>"
        reporte_html += f"&nbsp;&nbsp; Grandes: {self.total_dia_conteo.get('Grande', 0)}<br>"
        reporte_html += f"&nbsp;&nbsp; Medianas: {self.total_dia_conteo.get('Mediana', 0)}<br>"
        reporte_html += f"&nbsp;&nbsp; Pequeñas: {self.total_dia_conteo.get('Pequena', 0)}"
        
        self.total_display.setText(reporte_html)

    def exportar_y_reiniciar(self):
        # --- CORRECCIÓN: Calcular el total sumando las categorías de madurez ---
        total_manzanas = self.total_dia_conteo.get('madura', 0) + self.total_dia_conteo.get('intermedia', 0) + self.total_dia_conteo.get('verde', 0)

        if total_manzanas == 0:
            print("No hay nada que exportar.")
            self.total_display.setText("Nada que exportar. Contadores reiniciados.")
            return

        nombre_archivo = 'reporte_inventario.csv'
        
        datos_fila = [
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            total_manzanas,
            self.total_dia_conteo.get('madura', 0),
            self.total_dia_conteo.get('intermedia', 0),
            self.total_dia_conteo.get('verde', 0),
            self.total_dia_conteo.get('Grande', 0),
            self.total_dia_conteo.get('Mediana', 0),
            self.total_dia_conteo.get('Pequena', 0),
            f"{self.total_dia_valor:.2f}"
        ]
        
        try:
            archivo_existe = os.path.isfile(nombre_archivo)
            with open(nombre_archivo, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not archivo_existe:
                    writer.writerow(["Fecha y Hora", "Total Manzanas", "Maduras", "Intermedias", "Verdes", "Grandes", "Medianas", "Pequeñas", "Valor Total (S/)"])
                writer.writerow(datos_fila)
            
            print(f"Reporte guardado en {nombre_archivo}")
            self.total_display.setText(f"Reporte guardado en {nombre_archivo}.\n¡Contadores reiniciados!")
            
            self.total_dia_conteo.clear()
            self.total_dia_valor = 0.0
            
        except Exception as e:
            print(f"Error al exportar: {e}")
            self.total_display.setText(f"Error al guardar: {e}")

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return

        output_frame = frame.copy()
        h, w, _ = frame.shape
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
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
        lote_conteo_temp = Counter()
        lote_valor_temp = 0.0
        p_madura, p_intermedia, p_verde = self.get_precios()
        
        for (objectID, centroid) in objects.items():
            for rect in rects:
                x, y, w_box, h_box = rect
                if centroid[0] > x and centroid[0] < x + w_box and centroid[1] > y and centroid[1] < y + h_box:
                    x, y = max(0, x), max(0, y)
                    
                    roi_hsv_mask = hsv_frame[y:y+h_box, x:x+w_box]; mask_roi = cv2.inRange(roi_hsv_mask, np.array([0, 40, 40]), np.array([180, 255, 255])); contornos_roi, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if not contornos_roi: continue
                    contorno_manzana = max(contornos_roi, key=cv2.contourArea); ((cx, cy), radio) = cv2.minEnclosingCircle(contorno_manzana); diametro_real_mm = (radio * 2) / PIXELES_POR_MM
                    if objectID not in self.medidas_tamano: self.medidas_tamano[objectID] = deque(maxlen=10)
                    self.medidas_tamano[objectID].append(diametro_real_mm); diametro_promedio_mm = np.mean(self.medidas_tamano[objectID])
                    clasificacion_tamano = "Grande"
                    if diametro_promedio_mm < MANZANA_PEQUENA_MM: clasificacion_tamano = "Pequena"
                    elif diametro_promedio_mm < MANZANA_MEDIANA_MM: clasificacion_tamano = "Mediana"
                    
                    mask_final = np.zeros(hsv_frame.shape[:2], dtype="uint8"); contorno_manzana[:, :, 0] += x; contorno_manzana[:, :, 1] += y; cv2.drawContours(mask_final, [contorno_manzana], -1, 255, -1)
                    mask_color_valido = cv2.inRange(hsv_frame, np.array([0, 70, 50]), np.array([180, 255, 255])); mask_final = cv2.bitwise_and(mask_final, mask_color_valido)
                    hue_promedio = -1
                    if np.any(mask_final): hue_promedio = cv2.mean(hsv_frame, mask=mask_final)[0]
                    clasificacion_color = "No definido"
                    if 0 <= hue_promedio <= 18 or 170 <= hue_promedio <= 180: clasificacion_color = "Roja"
                    elif 35 <= hue_promedio <= 75: clasificacion_color = "Verde"
                    elif 19 <= hue_promedio <= 34: clasificacion_color = "Amarilla"
                    
                    clasificacion_madurez, confianza = "Analizando...", 0
                    if model_loaded:
                        roi = frame[y:y+h_box, x:x+w_box]
                        if roi.size > 0:
                            img_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH)); img_array = tf.keras.utils.img_to_array(img_resized); img_batch = np.expand_dims(img_array, axis=0)
                            prediction = classification_model.predict(img_batch, verbose=0); score = tf.nn.softmax(prediction[0])
                            pred_actual = CLASS_NAMES[np.argmax(score)]
                            if objectID not in self.predicciones_recientes: self.predicciones_recientes[objectID] = deque(maxlen=5)
                            self.predicciones_recientes[objectID].append(pred_actual)
                            if len(self.predicciones_recientes[objectID]) == 5:
                                clasificacion_madurez = max(set(self.predicciones_recientes[objectID]), key=list(self.predicciones_recientes[objectID]).count)
                                confianza = 100 * np.max(score)

                    if clasificacion_madurez != "Analizando...":
                        lote_conteo_temp.update([clasificacion_tamano, clasificacion_color, clasificacion_madurez])
                        if clasificacion_madurez == 'madura': lote_valor_temp += p_madura
                        elif clasificacion_madurez == 'intermedia': lote_valor_temp += p_intermedia
                        elif clasificacion_madurez == 'verde': lote_valor_temp += p_verde
                    
                    cv2.rectangle(output_frame, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"ID {objectID} ({clasificacion_madurez})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
        
        self.lote_actual_conteo = lote_conteo_temp
        self.lote_actual_valor = lote_valor_temp
        
        if objects:
            self.lote_display.setText(f"<b>Total Lote: {len(objects)} manzanas</b><br>"
                                      f"<b>Valor Lote: S/ {self.lote_actual_valor:.2f}</b><br><br>"
                                      f"Maduras: {self.lote_actual_conteo.get('madura', 0)} | "
                                      f"Intermedias: {self.lote_actual_conteo.get('intermedia', 0)} | "
                                      f"Verdes: {self.lote_actual_conteo.get('verde', 0)}")
        else:
            self.lote_display.setText("No se detectan manzanas")
        
        rgb_image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape; bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        self.timer.stop()
        self.cap.release()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = AsistenteInventarioApp()
    main_window.show()
    sys.exit(app.exec_())