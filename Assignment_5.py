import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from load_data import load_images


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


def plot_random_letter(images, pca):

    np.random.seed(seed = 123)
    index = np.random.randint(low = 0, high = len(images))
    pca_images = pca.transform(images)
    approximation = pca.inverse_transform(pca_images)

    plt.ion()
    plt.figure(figsize=(8,4))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(images[0].reshape(20,20), interpolation='nearest')
    plt.xlabel('%s components' % 400, fontsize = 10)
    plt.title('Original Image', fontsize = 14)

    # Principal components
    plt.subplot(1, 2, 2)
    plt.imshow(approximation[0].reshape(20, 20), interpolation='nearest')
    plt.xlabel('%s components' % pca.n_components_, fontsize = 10)
    plt.title('{} explained Variance'.format(pca.n_components), fontsize = 14)
    plt.show()
    return 0



# Step 1: load data
train_img, test_img, train_label, test_label = load_images()
# Step 2: standardize data to 0 mean and unit variance
train_img, test_img = standardize(train_img, test_img)
# Step 3: perform principal component analysis 
pca = decomposition.PCA(0.9)
pca.fit(train_img)
pca_train = pca.transform(train_img)
pca_test = pca.transform(test_img)
plot_random_letter(train_img, pca)

# logisticRegr = LogisticRegression(solver = 'lbfgs')
# logisticRegr.fit(train_img, train_label)
# logisticRegr.predict(test_img[0].reshape(1,-1))


