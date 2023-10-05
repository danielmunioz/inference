"""
Should be the same as 
https://github.com/unifyai/graph-compiler/blob/odsc/odsc/0_transpiling_functions/01_custom_composites/demo.py

"""
import sys
sys.path.append('..')

import torch
import kornia
import matplotlib.pyplot as plt
from demos.kornia.functions import dilate_edges, img_np, img
from graph_compiler import compile
from transpiler.transpiler import transpile


# show image
plt.axis('off')
plt.imsave("image.png", img_np)

dilated_edges = dilate_edges(img)
plt.imsave("dilated_edges_torch.png", dilated_edges[0, 0].numpy())
compiled = compile(dilate_edges, torch.randn_like(img))
compiled.show_graph()
compiled.function_frequencies()

# JAX
@transpile(source="torch", to="jax")
def dilate_edges(img):
    edges = kornia.filters.canny(img, hysteresis=False)[1]
    return kornia.morphology.dilation(edges, torch.ones(7, 7))


import jax
img = jax.numpy.array(img_np)
dilated_edges = dilate_edges(img)[1]
plt.imsave("dilated_edges_jax.png", dilated_edges[0, 0].numpy())
dilate_edges.show_graph()
dilate_edges.function_frequencies()

# TensorFlow
@transpile(source="torch", to="tensorflow")
def dilate_edges(img):
    edges = kornia.filters.canny(img, hysteresis=False)[1]
    return kornia.morphology.dilation(edges, torch.ones(7, 7))

import tensorflow as tf
img = tf.constant(img_np)
dilated_edges = dilate_edges(img)[1]
plt.imsave("dilated_edges_tf.png", dilated_edges[0, 0].numpy())
dilate_edges.show_graph()
dilate_edges.function_frequencies()

# NumPy
@transpile(source="torch", to="numpy")
def dilate_edges(img):
    edges = kornia.filters.canny(img, hysteresis=False)[1]
    return kornia.morphology.dilation(edges, torch.ones(7, 7))

dilated_edges = dilate_edges(img_np)[1]
plt.imsave("dilated_edges_np.png", dilated_edges[0, 0])
dilate_edges.show_graph()
dilate_edges.function_frequencies()