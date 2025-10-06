import sys
import cv2
import numpy as np
import tensorflow as tf
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QFrame, QCheckBox
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import QTimer, Qt
from collections import OrderedDict
from scipy.spatial import distance as dist

# (La clase CentroidTracker y la configuración inicial no cambian)
# ... (código del tracker, configuración de constantes y carga del modelo) ...
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared
    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
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
            for i in range(0, len(inputCentroids)): self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys()); objectCentroids = list(self.objects.values())
            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort(); cols = D.argmin(axis=1)[rows]
            usedRows = set(); usedCols = set()
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row); usedCols.add(col)
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.maxDisappeared: self.deregister(objectID)
            else:
                for col in unusedCols: self.register(inputCentroids[col])
        return self.objects

PIXELES_POR_MM = 3.2
MANZANA_PEQUENA_MM = 65
MANZANA_MEDIANA_MM = 80
model_loaded = False
try:
    model = tf.keras.models.load_model('modelo_manzanas.h5')
    CLASS_NAMES = ['intermedia', 'madura', 'verde']
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    model_loaded = True
except IOError:
    print("Error: No se encontró 'modelo_manzanas.h5'.")


class ClasificadorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracker = CentroidTracker()
        self.setWindowTitle("Clasificador de Manzanas")
        self.setGeometry(100, 100, 1000, 600)
        self.setStyleSheet("QWidget { background-color: #2E2E2E; color: #FFFFFF; } /* ... (resto del estilo) ... */")
        
        # --- NUEVO: Variables para controlar la visualización ---
        self.show_tamano = True
        self.show_color = True
        self.show_madurez = True
        
        # --- (La configuración del layout principal no cambia) ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.video_label = QLabel(self)
        main_layout.addWidget(self.video_label, 7)
        control_panel_layout = QVBoxLayout()
        main_layout.addLayout(control_panel_layout, 3)
        
        title_label = QLabel("Resultados de Clasificación")
        title_label.setFont(QFont("Arial", 16, QFont.Bold))
        control_panel_layout.addWidget(title_label)
        
        # --- NUEVO: Layout y Checkboxes para filtros ---
        checkbox_layout = QHBoxLayout()
        checkbox_layout.addWidget(QLabel("Mostrar:"))
        self.check_tamano = QCheckBox("Tamaño")
        self.check_tamano.setChecked(True)
        self.check_tamano.stateChanged.connect(self.update_display_flags)
        checkbox_layout.addWidget(self.check_tamano)

        self.check_color = QCheckBox("Color")
        self.check_color.setChecked(True)
        self.check_color.stateChanged.connect(self.update_display_flags)
        checkbox_layout.addWidget(self.check_color)
        
        self.check_madurez = QCheckBox("Madurez")
        self.check_madurez.setChecked(True)
        self.check_madurez.stateChanged.connect(self.update_display_flags)
        checkbox_layout.addWidget(self.check_madurez)
        
        control_panel_layout.addLayout(checkbox_layout)
        
        # (El resto del panel de control no cambia)
        line = QFrame(); line.setFrameShape(QFrame.HLine); control_panel_layout.addWidget(line)
        self.results_display = QLabel("Presiona 'Iniciar'"); self.results_display.setFont(QFont("Arial", 12)); self.results_display.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        control_panel_layout.addWidget(self.results_display)
        control_panel_layout.addStretch()
        button_layout = QHBoxLayout()
        self.toggle_button = QPushButton("Iniciar", self); self.toggle_button.clicked.connect(self.toggle_camera)
        self.quit_button = QPushButton("Salir", self); self.quit_button.clicked.connect(self.close)
        button_layout.addWidget(self.toggle_button); button_layout.addWidget(self.quit_button)
        control_panel_layout.addLayout(button_layout)
        
        self.cap = cv2.VideoCapture(0); self.timer = QTimer(); self.timer.timeout.connect(self.update_frame); self.is_running = False

    # --- NUEVA FUNCIÓN: Para actualizar las banderas de visualización ---
    def update_display_flags(self):
        self.show_tamano = self.check_tamano.isChecked()
        self.show_color = self.check_color.isChecked()
        self.show_madurez = self.check_madurez.isChecked()
        
    def toggle_camera(self):
        # (Sin cambios)
        if not self.is_running:
            self.timer.start(30); self.toggle_button.setText("Detener"); self.is_running = True
        else:
            self.timer.stop(); self.toggle_button.setText("Iniciar"); self.is_running = False

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret: return
        output_frame = frame.copy()
        
        # (El código de detección y seguimiento no cambia)
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        limite_bajo = np.array([0, 50, 50]); limite_alto = np.array([180, 255, 255])
        mascara = cv2.inRange(hsv_frame, limite_bajo, limite_alto)
        contornos, _ = cv2.findContours(mascara, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        rects_y_contornos = []
        for c in contornos:
            if cv2.contourArea(c) > 5000:
                x, y, w, h = cv2.boundingRect(c); rects_y_contornos.append(((x, y, x + w, y + h), c))
        rects = [rc[0] for rc in rects_y_contornos]
        objects = self.tracker.update(rects)
        
        all_results_text = []
        for (objectID, centroid) in objects.items():
            for (rect, contorno) in rects_y_contornos:
                (x, y, ex, ey) = rect
                if centroid[0] > x and centroid[0] < ex and centroid[1] > y and centroid[1] < ey:
                    w, h = ex - x, ey - y
                    # (Toda la lógica de clasificación no cambia)
                    ancho_real_mm = w / PIXELES_POR_MM
                    clasificacion_tamano = "Grande"
                    if ancho_real_mm < MANZANA_PEQUENA_MM: clasificacion_tamano = "Pequena"
                    elif ancho_real_mm < MANZANA_MEDIANA_MM: clasificacion_tamano = "Mediana"
                    mascara_color = np.zeros(frame.shape[:2], dtype="uint8")
                    cv2.drawContours(mascara_color, [contorno], -1, 255, -1)
                    hue = cv2.mean(hsv_frame, mask=mascara_color)[0]
                    clasificacion_color = "No definido"
                    if 0 <= hue <= 15: clasificacion_color = "Roja"
                    elif 38 <= hue <= 50: clasificacion_color = "Verde"
                    elif 20 <= hue <= 35: clasificacion_color = "Amarilla"
                    clasificacion_madurez = "N/A"; confianza = 0
                    if model_loaded:
                        roi = frame[y:ey, x:ex]; img_resized = cv2.resize(roi, (IMG_HEIGHT, IMG_WIDTH))
                        img_array = tf.keras.utils.img_to_array(img_resized); img_batch = np.expand_dims(img_array, axis=0)
                        prediction = model.predict(img_batch, verbose=0); score = tf.nn.softmax(prediction[0])
                        clasificacion_madurez = CLASS_NAMES[np.argmax(score)]; confianza = 100 * np.max(score)

                    # --- NUEVO: Construcción dinámica del texto ---
                    resultado_partes = [f"<b>Manzana {objectID}</b>"]
                    if self.show_tamano:
                        resultado_partes.append(f"&nbsp;&nbsp;Tamaño: {clasificacion_tamano} ({ancho_real_mm:.1f} mm)")
                    if self.show_color:
                        resultado_partes.append(f"&nbsp;&nbsp;Color: {clasificacion_color}")
                    if self.show_madurez and model_loaded:
                        resultado_partes.append(f"&nbsp;&nbsp;Madurez: {clasificacion_madurez} ({confianza:.1f}%)")
                    
                    all_results_text.append("<br>".join(resultado_partes))
                    
                    cv2.rectangle(output_frame, (x, y), (ex, ey), (0, 255, 0), 2)
                    cv2.putText(output_frame, f"ID: {objectID}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
        
        if all_results_text: self.results_display.setText("<br><br>".join(all_results_text))
        else: self.results_display.setText("No se detectan manzanas")
        
        # (El resto no cambia)
        rgb_image = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB); h, w, ch = rgb_image.shape; bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(self.video_label.width(), self.video_label.height(), Qt.KeepAspectRatio))

    def closeEvent(self, event):
        # (Sin cambios)
        self.timer.stop(); self.cap.release(); super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv); main_window = ClasificadorApp(); main_window.show(); sys.exit(app.exec_())