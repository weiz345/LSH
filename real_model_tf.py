import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image

class RealModelTF:
    def __init__(self):
        self.model = MobileNetV2(include_top=False, pooling='avg')

    def __call__(self, img):
        if img.shape != (224, 224, 3):
            img = image.smart_resize(img, (224, 224))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        emb = self.model.predict(img, verbose=0)
        return emb.flatten()
