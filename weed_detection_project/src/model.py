"""
Model architectures: custom CNN and transfer-learning backbones.
"""

from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, VGG16, ResNet50

INPUT_SHAPE = (224, 224, 3)


def build_custom_cnn(input_shape=INPUT_SHAPE):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid"),
    ], name="custom_weed_cnn")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_transfer_model(backbone_name="mobilenet", input_shape=INPUT_SHAPE, fine_tune=False):
    backbone_name = backbone_name.lower()

    if backbone_name == "mobilenet":
        base = MobileNetV2(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == "vgg16":
        base = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    elif backbone_name == "resnet50":
        base = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        raise ValueError(f"Unknown backbone: {backbone_name}")

    base.trainable = fine_tune

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid"),
    ], name=f"{backbone_name}_weed_classifier")

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def get_model(model_type="mobilenet"):
    if model_type == "custom":
        return build_custom_cnn()
    return build_transfer_model(backbone_name=model_type)


if __name__ == "__main__":
    m = get_model("custom")
    m.summary()
