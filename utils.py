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
            i = abc.find(str(pred[0]))
            j = abc.find(str(expe[0]))
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
