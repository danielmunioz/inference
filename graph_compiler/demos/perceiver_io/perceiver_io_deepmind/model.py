import cv2
import imageio
import urllib.request
import pickle
import jax.numpy as jnp
import jax
import numpy as np
import haiku as hk
import os
from demos.perceiver_io.perceiver_io_deepmind.imagenet_labels import IMAGENET_LABELS
from demos.perceiver_io.perceiver_io_deepmind import perceiver, io_processors


# Code taken from DeepMind's colab:
# https://github.com/deepmind/deepmind-research/blob/master/perceiver/colabs/imagenet_classification.ipynb

IMAGE_SIZE = (224, 224)

fourier_pos_configs = dict(
    input_preprocessor=dict(
        position_encoding_type="fourier",
        fourier_position_encoding_kwargs=dict(
            concat_pos=True, max_resolution=(224, 224), num_bands=64, sine_only=False
        ),
        prep_type="pixels",
        spatial_downsample=1,
    ),
    encoder=dict(
        cross_attend_widening_factor=1,
        cross_attention_shape_for_attn="kv",
        dropout_prob=0,
        num_blocks=8,
        num_cross_attend_heads=1,
        num_self_attend_heads=8,
        num_self_attends_per_block=6,
        num_z_channels=1024,
        self_attend_widening_factor=1,
        use_query_residual=True,
        z_index_dim=512,
        z_pos_enc_init_scale=0.02,
    ),
    decoder=dict(
        num_z_channels=1024,
        position_encoding_type="trainable",
        trainable_position_encoding_kwargs=dict(
            init_scale=0.02,
            num_channels=1024,
        ),
        use_query_residual=True,
    ),
)


def imagenet_classifier(config, images):
    input_preprocessor = io_processors.ImagePreprocessor(**config["input_preprocessor"])
    encoder = perceiver.PerceiverEncoder(**config["encoder"])
    decoder = perceiver.ClassificationDecoder(1000, **config["decoder"])
    model = perceiver.Perceiver(
        encoder=encoder, decoder=decoder, input_preprocessor=input_preprocessor
    )
    logits = model(images, is_training=False)
    return logits


imagenet_classifier = hk.transform_with_state(imagenet_classifier)

rng = jax.random.PRNGKey(4)
cwd = os.getcwd()
checkpoint_path = os.path.join(cwd, "demos", "imagenet_checkpoint.pystate")
if not os.path.exists(checkpoint_path):
    url = "https://storage.googleapis.com/perceiver_io/imagenet_fourier_position_encoding.pystate"
    _ = urllib.request.urlretrieve(url, checkpoint_path)

with open(checkpoint_path, "rb") as f:
    ckpt = pickle.loads(f.read())
params = ckpt["params"]
state = ckpt["state"]

image_path = os.path.join(cwd, "demos", "dog.jpg")
if not os.path.exists(image_path):
    url = "https://storage.googleapis.com/perceiver_io/dalmation.jpg"
    _ = urllib.request.urlretrieve(url, image_path)

with open(image_path, "rb") as f:
    img = imageio.imread(f)

MEAN_RGB = (0.485 * 255, 0.456 * 255, 0.406 * 255)
STDDEV_RGB = (0.229 * 255, 0.224 * 255, 0.225 * 255)


def normalize(im):
    return (im - np.array(MEAN_RGB)) / np.array(STDDEV_RGB)


def resize_and_center_crop(image):
    """Crops to center of image with padding then scales."""
    shape = image.shape
    image_height = shape[0]
    image_width = shape[1]
    padded_center_crop_size = (
        (224 / (224 + 32)) * np.minimum(image_height, image_width).astype(np.float32)
    ).astype(np.int32)
    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = [
        offset_height,
        offset_width,
        padded_center_crop_size,
        padded_center_crop_size,
    ]
    # image = tf.image.crop_to_bounding_box(image_bytes, *crop_window)
    image = image[
        crop_window[0] : crop_window[0] + crop_window[2],
        crop_window[1] : crop_window[1] + crop_window[3],
    ]
    return cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)


img = resize_and_center_crop(img)
img = normalize(img)
img = jnp.array(img)[None]

noise = jax.random.uniform(rng, (1, 224, 224, 3))

# Functions #
# --------- #


def imagenet_classify(image, override_encoder_config=None):
    # any k,v pairs in override_encode_config will replace those in fourier_pos_configs["encoder"]
    if override_encoder_config:
        for k, v in override_encoder_config.items():
            fourier_pos_configs["encoder"][k] = v
    return imagenet_classifier.apply(params, state, rng, fourier_pos_configs, image)


def print_labels(probs, indices):
    for i in list(indices):
        print(f"{IMAGENET_LABELS[i].split(',')[0][:15].ljust(20)}: {probs[i]}")
