import progressbar
import random
import pandas as pd
from PIL import Image
import numpy as np


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar


def load_image_dataset(csv_file_path='../../data_generation/field_image_data/a_list_of_data.csv',
                       desired_height=None,
                       desired_width=None,
                       value_range=None,
                       max_images=None,
                       force_grayscale=True):
     
    data_directory=csv_file_path[:csv_file_path.rfind('/')+1]
    data=pd.read_csv(csv_file_path) 
    image_paths=data_directory+data.image_file_name
    coordinates=np.asarray([data.x_shift, data.y_shift]).transpose()
    
    limit_msg = ''
    '''
    if max_images is not None and len(image_paths) > max_images:
        image_paths = random.sample(image_paths, max_images)
        limit_msg = " (limited to %d images by command line argument)" % (max_images,)
    '''
    
    print("Found %d images in %s%s." % (len(image_paths), data_directory, limit_msg))

    pb = create_progress_bar("Loading dataset ")


    storage = None

    image_idx = 0
    for fname in pb(image_paths):
        image = Image.open(fname)
        width, height = image.size
        if desired_height is not None and desired_width is not None:
            if width != desired_width or height != desired_height:
                image = image.resize((desired_width, desired_height), Image.BILINEAR)
        else:
            desired_height = height
            desired_width = width

        if force_grayscale:
            image = image.convert("L")

        img = np.array(image)

        if len(img.shape) == 2:
            # extra channel for grayscale images
            img = img[:, :, None]

        if storage is None:
            storage = np.empty((len(image_paths), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)

        storage[image_idx] = img

        image_idx += 1

    if value_range is not None:
        storage = (
            value_range[0] + (storage / 255.0) * (value_range[1] - value_range[0])
        )
    print("dataset loaded.", flush=True)
    num_samples=len(coordinates)
    train_samples=int(num_samples*3/4)
    
    return (storage[0:train_samples], coordinates[0:train_samples]), (storage[train_samples:], coordinates[train_samples:])
    
    
if __name__=="__main__":
    #print(latest_used_name('/net/hciserver03/storage/vdixit/GAN/tensorflow-infogan-encoded/tensorflow-infogan/PAW_ckpoint/gan'))
    csv_file_path='../../data_generation/field_image_data/a_list_of_data.csv'    
    (x_train, y_train), (x_test, y_test)=load_image_dataset(csv_file_path)
    print(x_train.shape, y_train.shape, x_test.shape)