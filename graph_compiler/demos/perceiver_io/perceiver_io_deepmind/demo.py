import sys
sys.path.append('..')

import ivy
import torch
import jax
import tensorflow as tf
import numpy as np
from graph_compiler import compile
from transpiler.transpiler import transpile
from demos.perceiver_io.perceiver_io_deepmind.model import imagenet_classify, img, print_labels, noise


# Compile PerceiverIO #
# ------------------- #

# perform inference on `img` with deepmind perceiver

logits, _ = imagenet_classify(img)
_, indices = jax.lax.top_k(logits[0], 5)
probs = jax.nn.softmax(logits[0])
print("Original- Top 5 labels:")
print_labels(probs, indices)


# compile deepmind perceiver, then perform inference

ivy.set_backend("jax")
comp_function = compile(imagenet_classify, args=(noise,))
comp_logits, _ = comp_function(img)
_, indices = jax.lax.top_k(comp_logits[0], 5)
probs = jax.nn.softmax(comp_logits[0])
print(
    f"Original implementation contains {len(comp_function._functions)} functions"
)
print("Compiled- Top 5 labels:")
print_labels(probs, indices)


# Transpile PerceiverIO #
# --------------------- #

np_classify = transpile(imagenet_classify, source="jax", to="numpy", args=(noise,))

torch_classify = transpile(
    imagenet_classify, source="jax", to="torch", args=(noise,)
)  # immediate transpile


@transpile(source="jax", to="tensorflow")  # decorated lazy
def tensorflow_classify(image):
    return imagenet_classify(image)


jax_classify = transpile(imagenet_classify, source="jax", to="jax")  # lazy transpile
jax_classify(noise)  # resolve transpile

transpiled_logits, _ = np_classify(img)
indices = np.argpartition(transpiled_logits[0], -5)[-5:]
probs = np.exp(transpiled_logits[0])/sum(np.exp(transpiled_logits[0]))
print(f"numpy graph contains {np_classify.graph_size} functions")
print("Transpiled numpy- Top 5 labels:")
print_labels(probs, reversed(list(indices)))

transpiled_logits, _ = torch_classify(img)
_, indices = torch.topk(transpiled_logits, 5)
probs = torch.nn.functional.softmax(transpiled_logits[0])
print(f"torch graph contains {torch_classify.graph_size} functions")
print("Transpiled torch - Top 5 labels:")
print_labels(probs, indices[0])

transpiled_logits, _ = tensorflow_classify(img)
_, indices = tf.math.top_k(transpiled_logits, 5)
indices = indices[0].numpy()
probs = tf.keras.activations.softmax(transpiled_logits)[0].numpy()
print(f"tensorflow graph contains {tensorflow_classify.graph_size} functions")
print("Transpiled tensorflow - Top 5 labels:")
print_labels(probs, indices)

transpiled_logits, _ = jax_classify(img)
_, indices = jax.lax.top_k(transpiled_logits[0], 5)
probs = jax.nn.softmax(transpiled_logits[0])
print(f"jax graph contains {jax_classify.graph_size} functions")
print("Transpiled jax - Top 5 labels:")
print_labels(probs, indices)
