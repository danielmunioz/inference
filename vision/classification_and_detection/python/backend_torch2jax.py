import jax
import jax.numpy as jnp
import haiku as hk
import numpy as np
import torch
import torchvision

import ivy
import backend


class BackendTorch2Jax(backend.Backend):
    def __init__(self):
        super(BackendTorch2Jax, self).__init__()
        self.sess = None
        self.model = None

    def version(self):
        return jax.__version__

    def name(self):
        return "torch2jax"

    def image_format(self):
        return "NCHW"

    def haiku_model_predict(self,image):
        module = self.model()
        out = module(image)
        return out

    def load(self, model_path, inputs=None, outputs=None):
        self.model = torchvision.models.__dict__["resnet50"](pretrained=False)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        if inputs:
            self.inputs = inputs
        else:
            self.inputs = []
            initializers = set()
            for i in self.model.graph.initializer:
                initializers.add(i.name)
            for i in self.model.graph.input:
                if i.name not in initializers:
                    self.inputs.append(i.name)
        # find outputs from the model if not passed in by config
        if outputs:
            self.outputs = outputs
        else:
            self.outputs = []
            for i in self.model.graph.output:
                self.outputs.append(i.name)
        
        # transpilation
        noise = torch.randn(1, 3, 224, 224)
        self.model = ivy.transpile(self.model, to="haiku", args=(noise,))

        rng_key = jax.random.PRNGKey(42)
        noise_image = jax.random.normal(key=rng_key, shape=(1, 3, 224, 224))
        self.transform_func = jax.jit(self.haiku_model_predict)
        self.transform_func = hk.transform(self.transform_func)
        self.params = self.transform_func.init(rng=rng_key, image=noise_image)

        return self


    def predict(self, feed):
        key = [key for key in feed.keys()][0]
        feed[key] = jnp.array(feed[key]).astype(float)
        output = self.transform_func.apply(rng=None, params=self.params, **feed)
        return np.array(output)[None,]