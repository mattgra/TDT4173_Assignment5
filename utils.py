import numpy as np
import matplotlib.pyplot as plt
from load_data import load_images


def conf_mat(predicted, expected, title = 'Confusion Matrix', return_matrix = False):

      N_classes = 26
      abc = 'abcdefghijklmnopqrstuvwxyz'
      confusion_matrix = np.zeros(shape=(N_classes, N_classes)).astype(np.uint8)
      
      # inconsistend format of predictions of different models (logistic regression, knn, ...)
      predicted.reshape(len(predicted),1)
      expected.reshape(len(expected), 1) 

      for pred, expe in zip(predicted, expected):
            try:
                  i = abc.find(str(pred[0]))
                  j = abc.find(str(expe[0]))
            except IndexError:
                  i, j = int(pred), int(expe)
            confusion_matrix[j, i] += 1

      fig = plt.figure()
      ax = fig.add_subplot(111)
      ax.set_aspect(1)
      res = ax.imshow(confusion_matrix, interpolation = 'nearest')

      width, height = confusion_matrix.shape
      
      false_pos_or_false_neg = np.copy(confusion_matrix)
      np.fill_diagonal(false_pos_or_false_neg, val = 0)  # values that are not on the diagonal are false positives/negatives

      # for x in range(width):
      #       for y in range(height):
      #             if false_pos_or_false_neg[x,y]:
      #                   color = 'red'
      #             else: color = 'black'
      #             ax.annotate(str(confusion_matrix[x,y]), xy=(y, x), color = color,
      #                         horizontalalignment='center',
      #                         verticalalignment='center')

      cb = fig.colorbar(res)
      plt.xlabel('Predicted')
      plt.ylabel('Expected')
      plt.xticks(range(width), abc)
      plt.yticks(range(height), abc)
      plt.title(title)
      plt.ion()
      plt.show()

      if return_matrix:
            return confusion_matrix
      else:
            return 0


def mean_and_std_plots(img_orig, img_preprocessed, prefix = 'Preprocessed'):

      plt.figure()
      means_orig = img_orig.mean(axis = 1)
      stds_orig = img_orig.std(axis = 1)
      means_preproc = img_preprocessed.mean(axis = 1)
      stds_preproc = img_preprocessed.std(axis = 1)

      plt.subplot(2,2,1)
      plt.hist(means_orig, bins = 100)
      plt.title('Original Mean Values')

      plt.subplot(2,2,2)
      plt.hist(stds_orig, bins = 100)
      plt.title('Original Standard Deviations')

      plt.subplot(2,2,3)
      plt.hist(means_preproc, bins = 100)
      plt.title(prefix + ' Mean Values')

      plt.subplot(2,2,4)
      plt.hist(stds_preproc, bins = 100)
      plt.title(prefix + ' Standard Deviations')

      plt.tight_layout()
      plt.show()


def plot_random_letters(images, labels, window_size = 4):

      ABC = 'abcdefghijklmnopqrstuvwxyz'
      N_IMAGES = len(images)
      WINDOW_SIZE = window_size  # number of images per row/col
      indices = np.random.randint(low = 0, high = N_IMAGES, size = WINDOW_SIZE*WINDOW_SIZE)
      plt.figure()
      plt.ion()
      
      for i, index in enumerate(indices):
            plt.subplot(WINDOW_SIZE, WINDOW_SIZE, i+1)
            plt.imshow(images[int(index)].reshape(20,20))
            letter = ABC[int(labels[index])]
            plt.title('Label: {}'.format(letter))
      
      plt.tight_layout()
      plt.show()
      return 0


def main():
      """Test"""

      from sklearn.linear_model import LogisticRegression
      train_img, test_img, train_label, test_label = load_images()
      logisticRegr = LogisticRegression(solver = 'lbfgs')
      logisticRegr.fit(train_img, train_label)
      prediction = logisticRegr.predict(test_img[0:100])
      actual = test_label[0:100]

      cf = conf_mat(predicted = prediction, expected = actual, return_matrix=True)

if __name__ == '__main__':
      main()
