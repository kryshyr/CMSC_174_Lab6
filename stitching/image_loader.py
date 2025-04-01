import os

import cv2


def load_images(directory = "data"):
    
    """ load images from the specified directory and return them as a list """
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist.")

    image_files = sorted([f for f in os.listdir(directory) if f.lower().endswith(('.jpg'))])
    images = [cv2.imread(os.path.join(directory, img_file)) for img_file in image_files]

    return images
