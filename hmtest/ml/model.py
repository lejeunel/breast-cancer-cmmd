def make_model(input_shape: tuple[int], dropout_rate: float = 0.0):
    from keras.applications import MobileNetV3Small, MobileNetV3Large

    return MobileNetV3Large(
        input_shape=input_shape,
        alpha=1.0,
        dropout_rate=dropout_rate,
        classes=1,
        minimalistic=True,
        include_preprocessing=True,
        include_top=True,
        weights=None,
        classifier_activation="sigmoid",
    )

    # return EfficientNetB0(
    #     input_shape=input_shape,
    #     classes=1,
    #     include_top=True,
    #     weights=None,
    #     classifier_activation="sigmoid",
    # )
