from os.path import join
import os


cwd = os.getcwd()

data_path = join(cwd, "data")


# We want more of that type of image
data_existing = join(data_path, "existing")


# There you have more of that type of image
data_generated = join(data_path, "generated")

log_dir = join(cwd, "logs")

image_save_dir = join(cwd, "generated_images")