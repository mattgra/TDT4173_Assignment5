import keras
import numpy as np
from load_data import load_images
from CNN import load_model
import skimage.io
import matplotlib.pyplot as plt
from matplotlib import patches


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def predict_boxes(image, stepSize, windowSize):
    boxes = []
    for (x, y, window) in sliding_window(image=image, stepSize=stepSize, windowSize=windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
            continue
        else:
            crop = window.reshape(1,20,20,1)
            p = loaded_model.predict(crop)
            boxes.append((x,y,p))

    return boxes


def draw_boxes(boxes, threshold):

    plt.ion()
    ABC = 'abcdefghijklmnopqrstuvqxyz'
    fig1 = plt.figure()
    ax = fig1.add_subplot(111,aspect='equal')
    ax.imshow(image)

    for x,y,p in boxes:
        if p.max() >= threshold:
            ax.add_patch(patches.Rectangle(
                (x,y),
                width = 20,
                height = 20,
                fill = False,
            ))
            ax.text(x+20, y,
            '%s, %.2f' % (ABC[p.argmax()],(p.max())),
            horizontalalignment='right',
            verticalalignment='bottom', fontsize=5) 
    plt.show()


file_path = '/Users/Matthias/Desktop/Exchange_Semester/Courses/ML and CB Reasoning/Assignments/Assignment_5/TDT4173_Assignment5/detection-images/detection-1.jpg'
image = np.array(skimage.io.imread(file_path))/255.
loaded_model = load_model()

STEPSIZE = 5
WINDOWSIZE = (20,20)
THRESHOLD = 0.99

boxes = predict_boxes(image=image, stepSize=STEPSIZE, windowSize=WINDOWSIZE)
draw_boxes(boxes, threshold = 0.99)

import tensorflow as tf
from tensorflow import image
box_origins = [[x,y] for x,y,_ in boxes]
box_ends = [[x+20, y+20] for x,y in box_origins]
scores = [s.max() for _,_,s in boxes]
reshaped_boxes = [[origin[1], origin[0], end[1], end[0]] for origin, end in zip(box_origins, box_ends)]

with tf.Session() as sess:
    selected_indices = image.non_max_suppression(reshaped_boxes, scores, max_output_size=len(scores))
    selected_boxes = tf.gather(reshaped_boxes, selected_indices)
