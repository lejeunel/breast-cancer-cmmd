import keras


def weighted_f1_scorer(threshold=0.5):
    return keras.metrics.F1Score(
        average=None,
        threshold=threshold,
    )
