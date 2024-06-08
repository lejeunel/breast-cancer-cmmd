from tensorflow.keras.models import Model


def make_model(input_shape: tuple[int], dropout_rate: float = 0.0):
    from keras.applications import MobileNetV3Large

    return MobileNetV3Large(
        input_shape=input_shape,
        alpha=0.75,
        dropout_rate=dropout_rate,
        classes=1,
        minimalistic=False,
        include_preprocessing=True,
        include_top=True,
        weights=None,
        classifier_activation="sigmoid",
    )
