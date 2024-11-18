import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers, preprocessing
import config
from datetime import datetime
from os.path import join
import wgan_gp_256

BATCH_SIZE = 8
IMG_SHAPE = (256, 256, 1)
NOISE_DIM = 128
EPOCHS = 1000

MODEL_NAME = "WGAN_256_aligned_decay"

LOG_DIR = join(config.log_dir, MODEL_NAME + "_" + str(datetime.now()))
MODEL_SAVE_PATH = os.path.join(config.cwd, "checkpoints", MODEL_NAME)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(config.log_dir, exist_ok=True)


def main():
    d_model = wgan_gp_256.get_discriminator_model(IMG_SHAPE)
    d_model.summary()
    g_model = wgan_gp_256.get_generator_model(NOISE_DIM)
    g_model.summary()
    
    
    initial_learning_rate = 0.0002
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=2500,
        decay_rate=0.96,
        staircase=True)
    generator_optimizer = keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.5, beta_2=0.9)
    discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

    wgan = wgan_gp_256.WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=3
    )
    
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        run_eagerly=False # for debugging = True
        )
    
    
    train_dataset = load_tf_dataset()
    callbacks = get_callbacks()
    
    wgan.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks)
    
    # Save the generator model after training
    g_model.save(f"{MODEL_NAME}_generator.keras")
    wgan.save(f"{MODEL_NAME}_full.keras")
    



def get_callbacks() -> list[keras.callbacks.Callback]:
    """
    Returns a list of usefull callbacks for monitoring GANs
    Logs images to tensorboard, logs the losses to tensorboard and saves modelcheckpoints

    Returns:
        list[keras.callbacks.Callback]: List of callbacks 
    """
    image_logger = GANMonitor(num_img=2, latent_dim=NOISE_DIM, log_dir=LOG_DIR, save_freq=20)
    tensorboard = keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    chkp = keras.callbacks.ModelCheckpoint(
        os.path.join(MODEL_SAVE_PATH, f"chk_{MODEL_NAME}_full.keras"),
        save_best_only=False,  
        save_weights_only=False,  
        save_freq='epoch' 
    )
    
    return [chkp, tensorboard, image_logger]
    
    
def load_tf_dataset() -> tf.data.Dataset:
    train_dataset = preprocessing.image_dataset_from_directory(
        os.path.join(os.getcwd(), "data", "aligned"),
        labels=None,
        label_mode=None,
        class_names=None,
        color_mode="grayscale",
        batch_size=None,
        image_size=IMG_SHAPE[0:2],
        shuffle=True,
        seed=None,
        validation_split=None,
        subset=None,
        crop_to_aspect_ratio=False,
        pad_to_aspect_ratio=False,
    )
    def normalize(train_images):
        return (train_images - 127.5) / 127.5  
        
    train_dataset = train_dataset.map(normalize).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset




"""
## Create a Keras callback that periodically saves generated images
"""

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=2, latent_dim=128, log_dir="", save_freq=20):
        self.num_img = num_img
        self.latent_dim = latent_dim
        self.log_dir = log_dir
        self.save_freq = save_freq  

        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_freq == 0:
            random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim), seed=42)
            generated_images = self.model.generator(random_latent_vectors)
            generated_images = (generated_images * 127.5) + 127.5  # Rescale to [0, 255]

            with self.file_writer.as_default():
                for i in range(self.num_img):
                    
                    # log to tensorboard
                    img = generated_images[i]
                    img = tf.image.convert_image_dtype(img, dtype=tf.uint8)  
                    img = tf.expand_dims(img, axis=0) 
                    tf.summary.image(f"generated_img_{i}", img, step=epoch)
                    
                    
                    if epoch % 50 == 0:
                        
                        img = generated_images[i].numpy()
                        img = keras.utils.array_to_img(img)
                        os.makedirs(config.data_generated, exist_ok=True)
                        img.save(os.path.join(config.data_generated,f"{MODEL_NAME}_{i}_{epoch}.png"))

            self.file_writer.flush()


if __name__ == "__main__":
    main()


