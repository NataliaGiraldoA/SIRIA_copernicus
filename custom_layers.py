import tensorflow as tf
from keras.saving import register_keras_serializable
from tensorflow.keras import layers

@register_keras_serializable()
class RandomSaturation(layers.Layer):
    def __init__(self, factor_min=0.8, factor_max=1.2, **kwargs):
        super().__init__(**kwargs)
        self.factor_min = factor_min
        self.factor_max = factor_max

    def call(self, inputs, training=None):
        if training:
            factor = tf.random.uniform([], self.factor_min, self.factor_max)
            return tf.image.adjust_saturation(inputs, factor)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'factor_min': self.factor_min,
            'factor_max': self.factor_max
        })
        return config

@register_keras_serializable()
class RandomHue(layers.Layer):
    def __init__(self, delta_min=-0.05, delta_max=0.05, **kwargs):
        super().__init__(**kwargs)
        self.delta_min = delta_min
        self.delta_max = delta_max

    def call(self, inputs, training=None):
        if training:
            delta = tf.random.uniform([], self.delta_min, self.delta_max)
            return tf.image.adjust_hue(inputs, delta)
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({
            'delta_min': self.delta_min,
            'delta_max': self.delta_max
        })
        return config
