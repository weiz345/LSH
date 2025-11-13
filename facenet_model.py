# tiny_facenet_model.py
import tensorflow as tf
import numpy as np

class TinyFaceNetModel:
    def __init__(self):
        base = tf.keras.applications.MobileNetV2(
            include_top=False, pooling="avg", input_shape=(160, 160, 3)
        )
        self.model = base

    def __call__(self, img):
        x = tf.image.resize(img, (160, 160))
        x = tf.expand_dims(x, 0)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
        emb = self.model(x)
        return emb.numpy().flatten()
