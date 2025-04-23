import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# CPU optimization
tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(2)

def create_data_pipeline():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.1
    )

    train_gen = train_datagen.flow_from_directory(
        'C:/Users/shaik/OneDrive/Desktop/plant D - Copy/Crop___Disease',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        'C:/Users/shaik/OneDrive/Desktop/plant D - Copy/Crop___Disease',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_gen, val_gen
def create_model(num_classes):
    base_model = applications.MobileNetV2(
        input_shape=(128, 128, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    base_model.trainable = True
    model = models.Sequential([
        base_model,
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
def train():
    train_gen, val_gen = create_data_pipeline()
    class_names = list(train_gen.class_indices.keys())
    model = create_model(len(class_names))
    model.compile(
        optimizer=optimizers.Adam(0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ModelCheckpoint('best_model.h5', save_best_only=True)
    ]
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        callbacks=callbacks
    )
    # Save model with class names metadata
    model.save('plant_model.h5')
    with open('class_names.txt', 'w') as f:
        f.write('\n'.join(class_names))
    
    print(f"Training complete. Classes: {class_names}")

if __name__ == '__main__':
    train()