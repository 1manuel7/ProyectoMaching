import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --- PARÁMETROS DE CONFIGURACIÓN ---
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
DATASET_DIR = 'dataset'
EPOCHS_FASE_1 = 15
EPOCHS_FASE_2 = 10 # Épocas adicionales para el ajuste fino
TOTAL_EPOCHS = EPOCHS_FASE_1 + EPOCHS_FASE_2

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

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print(f"Clases detectadas: {train_generator.class_indices}")

# --- CÁLCULO DE PESOS DE CLASE (CLASS WEIGHTS) ---
class_labels = list(train_generator.class_indices.keys())
class_indices = train_generator.classes
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(class_indices), y=class_indices)
class_weights_dict = dict(enumerate(class_weights))
print(f"Pesos de clase calculados: {class_weights_dict}")

# --- MODELO AVANZADO CON TRANSFER LEARNING ---
base_model = MobileNetV2(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), include_top=False, weights='imagenet')
base_model.trainable = False

inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# --- FASE 1: ENTRENAMIENTO INICIAL ---
print("\n--- INICIANDO FASE 1: ENTRENAMIENTO DE CAPAS SUPERIORES ---")
history = model.fit(
    train_generator,
    epochs=EPOCHS_FASE_1,
    validation_data=validation_generator,
    class_weight=class_weights_dict
)

# --- FASE 2: AJUSTE FINO (FINE-TUNING) ---
print("\n--- INICIANDO FASE 2: AJUSTE FINO (FINE-TUNING) ---")
base_model.trainable = True
for layer in base_model.layers[:-40]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# CORRECCIÓN: Se especifica el número total de épocas para que el entrenamiento continúe
history_fine = model.fit(
    train_generator,
    epochs=TOTAL_EPOCHS,
    validation_data=validation_generator,
    initial_epoch=history.epoch[-1] + 1, # Continuamos desde la siguiente a la última época
    class_weight=class_weights_dict
)

# Guardar el modelo final
model.save('modelo_manzanas.h5')
print("\n¡Entrenamiento con Ajuste Fino completado! Modelo guardado como 'modelo_manzanas.h5'")

# --- VISUALIZACIÓN DE RESULTADOS ---
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
loss = history.history['loss'] + history_fine.history['loss']
val_loss = history.history['val_loss'] + history_fine.history['val_loss']

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Precisión de Entrenamiento')
plt.plot(val_acc, label='Precisión de Validación')
plt.legend(loc='lower right')
plt.title('Precisión de Entrenamiento y Validación')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Pérdida de Entrenamiento')
plt.plot(val_loss, label='Pérdida de Validación')
plt.legend(loc='upper right')
plt.title('Pérdida de Entrenamiento y Validación')
plt.show()