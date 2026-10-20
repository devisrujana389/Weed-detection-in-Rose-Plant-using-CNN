"""
Dataset loading + augmentation pipeline.

Expects: data_dir/rose/*.jpg  and  data_dir/weed/*.jpg
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def get_data_generators(data_dir: str, img_size=IMG_SIZE, batch_size=BATCH_SIZE,
                         validation_split=0.2, seed=42):
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=30,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=validation_split,
    )

    train_gen = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", subset="training", seed=seed, shuffle=True,
    )

    val_gen = datagen.flow_from_directory(
        data_dir, target_size=img_size, batch_size=batch_size,
        class_mode="binary", subset="validation", seed=seed, shuffle=False,
    )

    return train_gen, val_gen


if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    train_gen, val_gen = get_data_generators(data_dir)
    print("Class indices:", train_gen.class_indices)
    print("Training samples:", train_gen.samples)
    print("Validation samples:", val_gen.samples)
