import tensorflow as tf
import numpy as np
import backend


class BackendTF2(backend.Backend):
    def __init__(self, compile=False):
        super(BackendTF2, self).__init__()
        self.compile = compile

    def version(self):
        return tf.__version__

    def name(self):
        return "tf2"

    def image_format(self):
        return "NHWC"
    
    def load(self, model_path, inputs=None, outputs=None):
        self.inputs, self.outputs = inputs, outputs
        self.model = tf.keras.applications.resnet50.ResNet50(
            include_top=True,weights=model_path,input_tensor=None,input_shape=None,
            pooling=None,classes=1000,
            )
        if self.compile:
            import ivy
            ivy.set_backend("tensorflow")
            noise = np.random.rand(1, 224, 224, 3)
            print("[+]Compiling TF2 model with Ivy...")
            self.model = ivy.compile(self.model, args=(noise,))
        return self

    def predict(self, feed):
        key = [key for key in feed.keys()][0]
        feed[key] = np.array(feed[key]).astype(float)
        output = self.model(feed[key]).numpy()[None,]
        return output