import os
import keras
import tensorflow as tf
import config
from d_utils import plot_multi
import matplotlib.pyplot as plt
from os.path import join
import numpy as np
import cv2


num_images = 50
LATENT_DIM = 128
RESOLUTION = 256

MODEL_SAVE_PATH = os.path.join(config.cwd, f"WGAN_{RESOLUTION}_generator.keras")
model = keras.models.load_model(MODEL_SAVE_PATH)

random_latent_vectors = tf.random.normal(shape=(num_images, LATENT_DIM))
generated  = model(random_latent_vectors)
generated = (generated * 127.5) + 127.5

images  = [gen.numpy() for gen in generated]

# plot_multi(images=images,
#            gray=True,
#            img_per_row=3,
#            main_title=f"Generated Images {RESOLUTION}",
#            show=True,
#            save_path=f"gen_{RESOLUTION}.png")


for i, im in enumerate(images):
    cv2.imwrite(join(config.data_generated, f"gen_{RESOLUTION}_{i}.png"), im)
