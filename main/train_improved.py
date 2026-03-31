import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight

# Constants
IMG_SIZE = (48, 48)
BATCH_SIZE = 32
NUM_CLASSES = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

class EmotionRecognitionModel:
    def __init__(self):
        self.model = None
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        
    def build_model(self, input_shape=(48, 48, 1)):
        """
        Build an improved CNN model with regularization and better architecture
        """
        model = models.Sequential([
            # First Convolution Block
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY), input_shape=input_shape),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            # Second Convolution Block
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.3),
            
            # Third Convolution Block
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.4),
            
            # Dense Layers
            layers.Flatten(),
            layers.Dense(512, kernel_regularizer=l2(WEIGHT_DECAY)),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.5),
            layers.Dense(NUM_CLASSES, activation='softmax')
        ])
        
        optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )
        
        self.model = model
        return model

    def create_data_generators(self):
        """
        Create data generators with augmentation for training
        """
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            zoom_range=0.15,
            brightness_range=[0.8, 1.2]
        )

        valid_datagen = ImageDataGenerator(rescale=1./255)
        
        return train_datagen, valid_datagen

def train_model():
    """
    Main training function with improved training process
    """
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(base_dir, 'Dataset_split', 'train')
    val_dir = os.path.join(base_dir, 'Dataset_split', 'validation')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    # Initialize model and data generators
    model = EmotionRecognitionModel()
    model.build_model()
    train_datagen, valid_datagen = model.create_data_generators()

    # Create generators
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = valid_datagen.flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    # Compute class weights
    labels = train_generator.classes
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weight_dict = dict(enumerate(class_weights))

    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(models_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        TensorBoard(
            log_dir=os.path.join(models_dir, 'logs'),
            histogram_freq=1
        )
    ]

    # Training parameters
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    validation_steps = validation_generator.samples // BATCH_SIZE
    epochs = 50

    print("\nStarting training with the following configuration:")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Learning rate: {LEARNING_RATE}")
    print("\nClass weights:")
    for emotion, weight in zip(model.emotions, class_weights):
        print(f"{emotion}: {weight:.2f}")

    # Train the model
    history = model.model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        class_weight=class_weight_dict,
        callbacks=callbacks
    )

    # Save training history
    history_path = os.path.join(models_dir, 'training_history.npy')
    np.save(history_path, history.history)
    print(f"\nTraining history saved to {history_path}")

if __name__ == "__main__":
    train_model()