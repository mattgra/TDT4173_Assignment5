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
    """Calculate class probability for each window returned by the sliding 
    window method and return boxes as an array of shape
    [box1 = (x1,y1,x2,y2,p), ...]"""

    boxes = []
    for (x, y, window) in sliding_window(image=image, stepSize=stepSize, windowSize=windowSize):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != windowSize[0] or window.shape[1] != windowSize[1]:
            continue
        else:
            crop = window.reshape(1,windowSize[0],windowSize[1],1)
            p = loaded_model.predict(crop)
            boxes.append((x,y,x+windowSize[0], y+windowSize[1],p))

    return boxes


def draw_boxes(image, boxes, threshold, title = 'Character Detection'):
    """Draw image with boxes that have a max class probability 
    above a threshold (> 0.9)"""

    plt.ion()
    ABC = 'abcdefghijklmnopqrstuvqxyz'
    fig1 = plt.figure()
    ax = fig1.add_subplot(111,aspect='equal')
    ax.imshow(image)

    for x,y,x2,y2,p in boxes:
        if p.max() >= threshold:
            ax.add_patch(patches.Rectangle(
                (x,y),
                width = x2-x,
                height = y2-y,
                fill = False,
            ))
            ax.text(x+x2-x, y,
            '%s, %.2f' % (ABC[p.argmax()],(p.max())),
            horizontalalignment='right',
            verticalalignment='bottom', fontsize=5) 
    
    plt.title(title)
    plt.show()
    return 0


def IoU(box1, box2):
    """Calculate Intersection over Union for a box shaped
    (x0,y0,x1,y1,p), where (x0,y0) - (x1,y1) spans the diagonal 
    of the box and p is the class probability array shaped (27,)"""

    box1 = box1[0:-1]  # last entry is class probability array
    box2 = box2[0:-1]
    
    max_window_size = np.max((box1,box2)) - np.min((box1,box2))
    mask1 = np.zeros(shape = (max_window_size,max_window_size))
    mask2 = np.zeros(shape = (max_window_size,max_window_size))

    x_orig_1, y_orig_1, x_end_1, y_end_1 = box1 - np.min((box1,box2))  # shift coordinate system
    x_orig_2, y_orig_2, x_end_2, y_end_2 = box2 - np.min((box1,box2))

    mask1[x_orig_1:x_end_1, y_orig_1:y_end_1] = 1
    mask2[x_orig_2:x_end_2, y_orig_2:y_end_2] = 1

    intersection = mask1 * mask2
    union = mask1 + mask2 - intersection + 1e-5
    return np.sum(intersection)/np.sum(union)


def non_maximum_suppression(boxes, IoU_threshold):
    """Takes array of boxes shaped as 
    [box1 = (x1,y1,x2,y2,p), box2 = (x1,y1,x2,y2,p), ...]
    and returns only the boxes with maximum p in a local area
    where IoU between boxes is larger than a threshold"""

    max_scores = np.array([box[-1].max() for box in boxes])
    rel_indices = np.where(max_scores >= THRESHOLD)[0]

    boxes_after_nms = []
    scores_after_nms = []
    for index1 in rel_indices:
        box = boxes[index1]
        max_score = max_scores[index1]

        for index2 in rel_indices:
            if index1 == index2:
                continue
            else:
                box2 = boxes[index2]
                score = max_scores[index2]
                iou = IoU(box, box2)
                if iou > IoU_threshold and score >= max_score:
                    box = box2
                    max_score = score
        
        if not box in boxes_after_nms:
            boxes_after_nms.append(box)
            scores_after_nms.append(max_score)
    
    return boxes_after_nms



if __name__ == '__main__':

    # load image    
    file_path = '/Users/Matthias/Desktop/Exchange_Semester/Courses/ML and CB Reasoning/Assignments/Assignment_5/TDT4173_Assignment5/detection-images/detection-1.jpg'
    image = np.array(skimage.io.imread(file_path))/255.
    loaded_model = load_model()

    # set params for detection/sliding window
    STEPSIZE = 4
    WINDOWSIZE = (20,20)
    THRESHOLD = 0.99

    # predict ALL boxes in the image
    boxes = predict_boxes(image=image, stepSize=STEPSIZE, windowSize=WINDOWSIZE)
    draw_boxes(image = image, boxes = boxes, threshold = THRESHOLD)

    # Non Maximum Suppression
    # TODO: now it's necessary to perform NMS 2 times due to the order of loop in the NMS method
    nms_boxes = non_maximum_suppression(boxes, IoU_threshold = 0.2)
    nms_boxes = non_maximum_suppression(nms_boxes, IoU_threshold = 0.2)
    draw_boxes(image = image, boxes = nms_boxes, threshold = THRESHOLD, title = 'After NMS')
