import keras.ops
from keras import Model, layers


def make_backbone(input_shape: tuple[int], dropout_rate: float):
    from keras.applications import MobileNetV3Small

    return MobileNetV3Small(
        input_shape=input_shape,
        alpha=1.0,
        dropout_rate=dropout_rate,
        minimalistic=False,
        include_preprocessing=True,
        include_top=False,
        weights=None,
    )


class BreastClassifier(Model):
    def __init__(
        self,
        input_shape: tuple[int],
        dropout_rate: float = 0.0,
        n_abnormalities=2,
        n_units=1280,
    ):
        super().__init__()
        self.backbone = make_backbone(input_shape, dropout_rate)
        self.global_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.upsampler = layers.Conv2D(
            filters=n_units,
            kernel_size=1,
            padding="same",
            use_bias=True,
            name="upsampler",
        )

        self.abnormality_clf = layers.Conv2D(
            filters=n_abnormalities,
            kernel_size=1,
            padding="same",
            name="abnormality_logits",
        )

        self.diagn_clf = layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding="same",
            name="diagn_logit",
        )

        self.distil_clf = layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding="same",
            name="distil_diagn_logit",
        )

        self.relu = layers.ReLU()
        self.batchnorm_bottleneck = layers.BatchNormalization()
        self.batchnorm_distil = layers.BatchNormalization()

    def call(self, images, *args, **kwargs) -> dict:
        x = self.backbone(images)
        x = self.global_pool(x)
        x = self.upsampler(x)
        x = self.batchnorm_bottleneck(x)
        x = self.relu(x)

        logit_diagn_pre = self.diagn_clf(x)
        logits_abnorm = self.abnormality_clf(x)

        merged = keras.ops.concatenate([logit_diagn_pre, logits_abnorm], axis=-1)
        merged = self.batchnorm_distil(merged)
        merged = self.relu(merged)

        logit_diagn_post = self.distil_clf(merged)

        return {
            "diagn_pre": keras.ops.squeeze(logit_diagn_pre)[..., None],
            "diagn_post": keras.ops.squeeze(logit_diagn_post)[..., None],
            "abnorm": keras.ops.squeeze(logits_abnorm),
        }
