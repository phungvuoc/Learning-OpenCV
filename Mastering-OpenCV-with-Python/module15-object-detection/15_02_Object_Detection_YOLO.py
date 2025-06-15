import cv2
import sys
import numpy as np
import requests
from os import path

# Initialize the parameters
objectnessThreshold = 0.5 # Objectness threshold, high values filter out low objectness
confThreshold = 0.5       # Confidence threshold, high values filter out low confidence detections
nmsThreshold = 0.4        # Non-maximum suppression threshold, higher values result in duplicate boxes per object 
inpWidth = 416            # Width of network's input image, larger is slower but more accurate
inpHeight = 416           # Height of network's input image, larger is slower but more accurate


# Load names of classes.
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov4.cfg"
modelWeights = "yolov4.weights"

# If model not present in the directory, download.
if(not path.exists(modelWeights)):

    url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights"

    r = requests.get(url)

    print('Downloading YOLO v4 Model.......')

    with open(modelWeights, 'wb') as f:
        f.write(r.content)

    print('\nyolov4.weights Download complete!')

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)

def getOutputsNames(net):
    """Get the names of all output layers in the network."""
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1


def display_objects(frame, outs):
    """Remove the bounding boxes with low confidence using non-maxima suppression."""
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    
    # Loop through all outputs.
    for out in outs:
        for detection in out:
            if detection[4] > objectnessThreshold:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 255, 255), 2)
        label = "{}:{:.2f}".format(classes[classIds[i]], confidences[i])
        display_text(frame, label, left, top)

def display_text(im, text, x, y):
    """Draw text onto image at location."""
    
    # Get text size 
    textSize = cv2.getTextSize(text, FONTFACE, FONT_SCALE, THICKNESS)
    dim = textSize[0]
    baseline = textSize[1]
            
    # Use text size to create a black rectangle. 
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(im, text, (x, y + dim[1]), FONTFACE, FONT_SCALE, (0, 255, 255), THICKNESS, cv2.LINE_AA)

# Inference using Yolo v4.
# Process inputs
imagePath = "traffic.jpg"
frame = cv2.imread(imagePath)

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

# Sets the input to the network
net.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net.forward(getOutputsNames(net))

# Remove the bounding boxes with low confidence
display_objects(frame, outs)

# Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
t, _ = net.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1, cv2.LINE_AA)
cv2.imshow('Detections', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Inference using YOLO v4 Tiny.
tiny_modelConfiguration = "yolov4-tiny.cfg"
tiny_modelWeights = "yolov4-tiny.weights"
    
if(not path.exists(tiny_modelWeights)):

    url = "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"

    r = requests.get(url)

    print('Downloading YOLO v4 Tiny Model.......')

    with open(tiny_modelWeights, 'wb') as f:
        f.write(r.content)

    print('\nyolov4-tiny.weights Download complete!!')
    
net_tiny = cv2.dnn.readNetFromDarknet(tiny_modelConfiguration, tiny_modelWeights)

# Process inputs
imagePath = "traffic.jpg"
frame = cv2.imread(imagePath)

# Create a 4D blob from a frame.
blob = cv2.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

# Sets the input to the network
net_tiny.setInput(blob)

# Runs the forward pass to get output of the output layers
outs = net_tiny.forward(getOutputsNames(net_tiny))

# Remove the bounding boxes with low confidence
display_objects(frame, outs)

# Put efficiency information. The function getPerfProfile returns the overall time for
# inference(t) and the timings for each of the layers(in layersTimes).
t, _ = net_tiny.getPerfProfile()
label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
cv2.putText(frame, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

cv2.imshow('Detections', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()