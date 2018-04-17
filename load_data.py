import numpy as np
import os
import skimage.io
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def load_images(flatten = True, test_size = 0.2):
    """Load images into following formats:
    - flatten = True:  images = (N_IMAGES, HEIGHT*WIDTH) = (7112, 400)
    - flatten = False: images = (N_IMAGES, HEIGHT, WIDTH) = (7112, 20, 20)
    - labels = (N_IMAGES,1) -> array of strings  TODO: evtl. change to int where 0 = 'a', ... 25 = 'z'?
    """

    PATH = 'chars74k-lite'
    HEIGHT, WIDTH = 20, 20
    N_FEATURES = HEIGHT*WIDTH
    N_IMAGES = 7112

    # create empty arrays for images and labels
    if flatten:
        images = np.zeros(shape = (N_IMAGES, N_FEATURES))
    else:
        images = np.zeros(shape = (N_IMAGES, HEIGHT, WIDTH))
    labels = np.zeros(shape = (N_IMAGES, 1)).astype(str)
    
    # load images and labels
    i = 0
    print('... loading images')
    for letter in tqdm('abcdefghijklmnopqrstuvwxyz'):
        for img_name in os.listdir(os.path.join(PATH,letter)):
            file_path = os.path.join(PATH,letter,img_name)
            if flatten:
                images[i] = np.array(skimage.io.imread(file_path)).flatten()
            else:
                images[i] = np.asarray(skimage.io.imread(file_path))
            labels[i] = letter
            i += 1

    # split into training and test data
    train_img, test_img, train_label, test_label = train_test_split(images, labels, test_size = test_size, random_state = 0)
    
    print('N_Train: %s \nN_Test: %s \nImage Shape: %s' % (len(train_img), len(test_img), train_img[0].shape))
    print('... loading finished!')

    return train_img, test_img, train_label, test_label
