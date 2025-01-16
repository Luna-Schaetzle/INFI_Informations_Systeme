# Bibliotheken importieren
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Add, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os

# Pfade definieren
base_dir = "/home/luna/5BHWII/INFI_Informations_Systeme/Aufgabe_6/img/"
image_size = (256, 256)  # Große Bildgröße für Details
batch_size = 64  # Größerer Batch für bessere GPU-Auslastung

# Datenaufbereitung
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# Funktion für Residual Blöcke (ResNet-Stil)
def residual_block(x, filters, kernel_size=(3, 3)):
    shortcut = Conv2D(filters, (1, 1), padding='same')(x)
    x = Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, kernel_size, padding='same', activation=None)(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = tf.keras.layers.ReLU()(x)
    return x

# CNN-Modell definieren
inputs = Input(shape=(image_size[0], image_size[1], 3))

# Erste Faltungsschicht mit großem Filter
x = Conv2D(64, (7, 7), strides=(2, 2), activation='relu', padding='same')(inputs)
x = BatchNormalization()(x)
x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

# Residual Blöcke
for filters in [64, 128, 256, 512]:
    for _ in range(3):
        x = residual_block(x, filters=filters)

# Global Average Pooling, um die Feature Maps zusammenzufassen
x = GlobalAveragePooling2D()(x)

# Vollständig verbundene Schichten für Klassifikation
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(3, activation='softmax')(x)  # 3 Klassen: Stein, Papier, Schere

# Modell kompilieren
model = Model(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks definieren
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_accuracy', mode='max')
]

# Modell trainieren
epochs = 50  # Mehr Epochen, da du Ressourcen hast
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=callbacks
)

# Speichern des endgültigen Modells
model.save('final_rock_paper_scissors_cnn.h5')


from tensorflow.keras.utils import plot_model

model = "rock_paper_scissors_cnn.h5"

plot_model(model, to_file='model_architecture.png', show_shapes=True)
