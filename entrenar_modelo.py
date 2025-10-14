import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, RandomFlip, RandomRotation, RandomZoom
import matplotlib.pyplot as plt

# --- PARÁMETROS DE CONFIGURACIÓN ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32 # Lotes de 32 imágenes por paso
DATASET_DIR = 'dataset' # Asegúrate que tu carpeta se llame así

# --- NUEVO: CONFIGURACIÓN DE AUMENTACIÓN DE DATOS ---
# ImageDataGenerator crea nuevas imágenes a partir de las existentes
# con pequeñas variaciones. Esto hace al modelo mucho más robusto.
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normaliza los píxeles de 0-255 a 0-1
    rotation_range=20,         # Rota la imagen hasta 20 grados
    width_shift_range=0.2,     # Desplaza la imagen horizontalmente
    height_shift_range=0.2,    # Desplaza la imagen verticalmente
    shear_range=0.2,           # Inclina la imagen
    zoom_range=0.2,            # Hace zoom a la imagen
    horizontal_flip=True,      # Invierte la imagen horizontalmente
    fill_mode='nearest',       # Rellena píxeles nuevos con los más cercanos
    validation_split=0.2       # Usa el 20% de los datos para validación
)

# --- CARGA DE DATOS DESDE LAS CARPETAS ---
# El generador cargará las imágenes, aplicará la aumentación y las dividirá.
train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Para clasificación multi-clase
    subset='training'         # Este es el conjunto de entrenamiento
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'       # Este es el conjunto de validación
)

# Obtener el número de clases automáticamente
num_classes = len(train_generator.class_indices)
print(f"Clases detectadas: {train_generator.class_indices}")


# --- CONSTRUCCIÓN DEL MODELO DE IA (CNN) ---
# Se mantiene la misma arquitectura, ya que es sólida
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5), # Capa para prevenir el sobreajuste
    Dense(num_classes, activation='softmax') # Softmax para clasificación multi-clase
])

# Compilar el modelo
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()


# --- ENTRENAMIENTO DEL MODELO ---
# Se aumenta el número de épocas porque la aumentación requiere más
# tiempo para que el modelo aprenda de todas las variaciones.
EPOCHS = 25 # Puedes probar con 25, 30 o incluso 50

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Guardar el modelo final
model.save('modelo_manzanas.h5')
print("\n¡Entrenamiento completado! El modelo ha sido guardado como 'modelo_manzanas.h5'")


# --- VISUALIZACIÓN DE RESULTADOS ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Precisión de Entrenamiento')
plt.plot(epochs_range, val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Pérdida de Entrenamiento')
plt.plot(epochs_range, val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()