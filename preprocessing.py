import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from load_data import load_images
from utils import mean_and_std_plots


def standardize(train_img, test_img, return_transform = False):
    """Standardize data to have 0 mean and unit variance"""

    scaler = StandardScaler()
    scaler.fit(train_img)  # fit on training set only!

    train_img = scaler.transform(train_img)
    test_img = scaler.transform(test_img)

    if return_transform:
        return train_img, test_img, scaler
    else:
        return train_img, test_img


def principal_components_transform(train_img, test_img, variance_explained = 1):
    """Calculate transformation to transform train_img to it's principal components and then
    apply it to train_img AND test_img"""

    if np.max(train_img) > 50 or np.mean(train_img) > 50:
        print('Warning: Data should be scaled before applying a PCA!')

    pca = decomposition.PCA(variance_explained)
    pca.fit(train_img)
    pca_train = pca.transform(train_img)
    pca_test = pca.transform(test_img)

    if variance_explained < 1. :
        print('Warning: PCA reduces dimensionality of feature space from {0} to {1} components'.format(400, pca.n_components_))

    return pca_train, pca_test



if __name__ == '__main__':

    train_img, test_img, train_label, test_label = load_images()


