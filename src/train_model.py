from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

# Paths for the data directories
train_data_dir = './data/train/train'  
test_data_dir = './data/test/test'     

# Data augmentation and splitting for training and validation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.1  # Reserve 10% of training data for validation
)

# Training data generator
train_generator = datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Specifies that this generator is for training data
)

# Validation data generator
validation_generator = datagen.flow_from_directory(
    directory=train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Specifies that this generator is for validation data
)

# Create model using InceptionV3 as the base
def create_model(input_shape, num_classes):
    base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=input_shape))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Number of classes
num_classes = len(train_generator.class_indices)

# Create model
model = create_model((224, 224, 3), num_classes=num_classes)

# Setup checkpointing
checkpoint_callback = ModelCheckpoint(
    'models/visionquest_model_best.h5',
    monitor='val_loss',
    save_best_only=True,
    verbose=1
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10,
    callbacks=[checkpoint_callback]
)
