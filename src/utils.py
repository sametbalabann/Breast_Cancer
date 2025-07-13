from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, img_size=(50, 50), batch_size=32):
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )

    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="training"
    )

    val_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="binary",
        subset="validation"
    )

    return train_gen, val_gen