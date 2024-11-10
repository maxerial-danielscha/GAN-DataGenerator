import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import tensorflow as tf
from keras import layers, preprocessing
import config


import wgan_gp

BATCH_SIZE = 8
IMG_SHAPE = (256, 256, 1)

NOISE_DIM = 128


EPOCHS = 100



IMAGE_SAVE_DIR = os.path.join(config.image_save_dir, "wgan_gp_128")
os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)



def main():
    
    d_model = wgan_gp.get_discriminator_model(IMG_SHAPE)
    d_model.summary()
    
    
    g_model = wgan_gp.get_generator_model(NOISE_DIM)
    g_model.summary()
    
    # Instantiate the optimizer for both networks
    # (learning_rate=0.0002, beta_1=0.5 are recommended)
    generator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    discriminator_optimizer = keras.optimizers.Adam(
        learning_rate=0.0002, beta_1=0.5, beta_2=0.9
    )
    
    # Define the loss functions for the discriminator,
    # which should be (fake_loss - real_loss).
    # We will add the gradient penalty later to this loss function.
    def discriminator_loss(real_img, fake_img):
        real_loss = tf.reduce_mean(real_img)
        fake_loss = tf.reduce_mean(fake_img)
        return fake_loss - real_loss
    
    
    # Define the loss functions for the generator.
    def generator_loss(fake_img):
        return -tf.reduce_mean(fake_img)
    
    os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    
    cbk = wgan_gp.GANMonitor(num_img=1, latent_dim=NOISE_DIM, save_dir=IMAGE_SAVE_DIR)
    tensorboard = keras.callbacks.TensorBoard(log_dir=config.log_dir)

    wgan = wgan_gp.WGAN(
        discriminator=d_model,
        generator=g_model,
        latent_dim=NOISE_DIM,
        discriminator_extra_steps=3,
    )
    
    
    wgan.compile(
        d_optimizer=discriminator_optimizer,
        g_optimizer=generator_optimizer,
        g_loss_fn=generator_loss,
        d_loss_fn=discriminator_loss,
        run_eagerly=False # for debugging = True
        )
    
    train_dataset = load_tf_dataset()
    
    
    wgan.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[cbk, tensorboard])
    


def load_tf_dataset():
    train_dataset = preprocessing.image_dataset_from_directory(
        config.data_path,
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
        return (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1], bc noise will also be in that range
        
    train_dataset = train_dataset.map(normalize).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
    
    return train_dataset


if __name__ == "__main__":
    main()
