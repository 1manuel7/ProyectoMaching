import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# --- CONFIGURACIÓN ---
DATASET_DIR = 'dataset'
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16 # Número de imágenes a procesar a la vez

# --- 1. CARGAR EL DATASET ---
# TensorFlow puede cargar imágenes directamente desde las carpetas que creaste
print("Cargando dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2, # Usamos el 20% de las imágenes para validar el modelo
    subset="training",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE)

# Guardamos los nombres de las categorías (madura, intermedia, verde)
class_names = train_dataset.class_names
print("Categorías encontradas:", class_names)


# --- 2. CONSTRUIR EL MODELO (TRANSFER LEARNING) ---
# Usaremos un modelo pre-entrenado (MobileNetV2) para obtener alta precisión
# [cite_start]con nuestro pequeño dataset. [cite: 129, 130]
base_model = tf.keras.applications.MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False # Congelamos el modelo base

# Creamos nuestro propio clasificador encima del modelo base
model = tf.keras.Sequential([
    # Capa para normalizar los pixeles de 0-255 a 0-1
    layers.Rescaling(1./255, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(len(class_names), activation='softmax') # Capa de salida con una neurona por categoría
])


# --- 3. COMPILAR Y ENTRENAR EL MODELO ---
print("Compilando el modelo...")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

print("Entrenando el modelo...")
# El entrenamiento puede tardar varios minutos dependiendo de tu computadora
epochs = 10
history = model.fit(
  train_dataset,
  validation_data=validation_dataset,
  epochs=epochs
)


# --- 4. GUARDAR EL MODELO ENTRENADO ---
print("Guardando el modelo entrenado...")
model.save('modelo_manzanas.h5')
print("¡Modelo guardado como 'modelo_manzanas.h5'!")