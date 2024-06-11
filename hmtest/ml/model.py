from tensorflow.keras import Model, layers


def make_model(input_shape: tuple[int], dropout_rate: float = 0.0):
    from tensorflow.keras.applications import MobileNetV3Small

    n_abnormalities = 2
    n_types = 1

    backbone = MobileNetV3Small(
        input_shape=input_shape,
        alpha=1.0,
        dropout_rate=dropout_rate,
        minimalistic=False,
        include_preprocessing=False,
        include_top=False,
        weights=None,
    )
    bb_inputs = backbone.inputs
    bb_outputs = backbone.outputs
    x = layers.GlobalAveragePooling2D(keepdims=True)(bb_outputs[0])
    feat_vec = layers.Dense(1280, activation="relu", name="feat_upsampler")(x)

    # abnormality classifier
    abnorm_out = layers.Dense(n_abnormalities, name="abnorm_clf")(feat_vec)
    abnorm_out = layers.Flatten()(abnorm_out)

    # tumor type classifier
    type_pre_out = layers.Dense(n_types, name="type_clf")(feat_vec)
    type_pre_out = layers.Flatten()(type_pre_out)

    # tumor type classifier with abnormality insight
    type_post_out = layers.Concatenate(axis=-1, name="type_abnorm_agg")(
        [abnorm_out, type_pre_out]
    )
    type_post_out = layers.Dense(16, activation="relu", name="type_hidden_unit")(
        type_post_out
    )
    type_post_out = layers.Dense(1, name="type_post_clf")(type_post_out)
    type_post_out = layers.Flatten()(type_post_out)

    breast_clf = Model(
        inputs=bb_inputs,
        outputs={
            "abnorm": abnorm_out,
            "type_pre": type_pre_out,
            "type_post": type_post_out,
        },
    )

    return breast_clf
