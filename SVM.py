import numpy as np
from load_data import load_images
from preprocessing import standardize, principal_components_transform
from sklearn import svm
from utils import conf_mat

train_img, test_img, train_label, test_label = load_images()
train_img_scaled, test_img_scaled = standardize(train_img, test_img)
clf = svm.SVC()
clf.fit(train_img_scaled, train_label)
score = clf.score(test_img_scaled, test_label)
prediction = clf.predict(test_img_scaled)
conf_mat(prediction, test_label)
print('Mean Accuracy for SVM with standardized images as input: {0}%'.format(np.round(score*100, decimals=2)))
