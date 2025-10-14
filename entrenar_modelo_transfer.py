import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# --- PARÁMETROS DE CONFIGURACIÓN ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATASET_DIR = 'dataset'

# --- AUMENTACIÓN DE DATOS ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(DATASET_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', subset='training')
validation_generator = train_datagen.flow_from_directory(DATASET_DIR, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode='categorical', subset='validation')

num_classes = len(train_generator.class_indices)
print(f"Clases detectadas: {train_generator.class_indices}")

# --- MODELO AVANZADO CON TRANSFER LEARNING ---
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False # Congelamos el modelo base inicialmente

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

# Compilación inicial con una tasa de aprendizaje estándar
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# --- FASE 1: ENTRENAMIENTO INICIAL ---
print("\n--- INICIANDO FASE 1: ENTRENAMIENTO DE CAPAS SUPERIORES ---")
history = model.fit(train_generator, epochs=10, validation_data=validation_generator)

# --- FASE 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- INICIANDO FASE 2: AJUSTE FINO (FINE-TUNING) ---")
base_model.trainable = True # Descongelamos el modelo base

# Solo re-entrenaremos las últimas capas del modelo base.
# Congelamos todo excepto las últimas 20 capas.
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Re-compilamos el modelo con una tasa de aprendizaje MUY BAJA.
# Esto es crucial para no destruir el conocimiento previo.
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continuamos el entrenamiento por unas pocas épocas más
history_fine = model.fit(train_generator, epochs=5, validation_data=validation_generator, initial_epoch=history.epoch[-1])

# Guardar el modelo final
model.save('modelo_manzanas.h5')
print("\n¡Entrenamiento con Ajuste Fino completado! Modelo guardado como 'modelo_manzanas.h5'")