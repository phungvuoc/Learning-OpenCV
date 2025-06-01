import cv2
import numpy as np
import glob

# Read Class File.
# Read the ImageNet class names.
with open('input/classification_classes_ILSVRC2012.txt', 'r') as f:
    image_net_names = f.read().split('\n')

# Save the names of all possible classifications, removing empty final line.
class_names = image_net_names[:-1]

# Verify the size, and inspect one of the classes by name.
print(len(class_names), class_names[0])

# Loading the Classification model.
config_file = 'models/DenseNet_121.prototxt'
model_file = 'models/DenseNet_121.caffemodel'

model = cv2.dnn.readNet(model=model_file, config=config_file, framework='Caffe')

# Load and display the image from disk.
tiger_img = cv2.imread('input/image1.jpg')
print('Press any key to continue')
cv2.imshow('Image', tiger_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Converting image to blob.
blob = cv2.dnn.blobFromImage(image=tiger_img, scalefactor=0.017, size=(224, 224), 
	mean=(104, 117, 123), swapRB=False, crop=False)

# Set the input blob for the neural network.
model.setInput(blob)

# Detections using the DNN Model.
# Pass the blob forward through the network.
outputs = model.forward()
final_outputs = outputs[0]

# Make all the outputs 1D, where each represents likihood of matching one of the 1K classification groups.
final_outputs = final_outputs.reshape(1000, 1)

# Get the class label index with the max confidence.
label_id = np.argmax(final_outputs)

# Convert score to probabilities for all matches.
probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))

print(probs[:10])
print("Max probability:", np.max(probs))

# Get the final highest probability
final_prob = np.max(probs) * 100.0

# Map the max confidence to the class label names.
out_name = class_names[label_id]
out_text = f"{out_name}, {final_prob:.3f}%"

# Display the image, best matched classification, and confidence.
cv2.imshow(str(out_text), tiger_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Some more examples.
def classify(img):
    image = img.copy()
    # Create blob from image.
    blob = cv2.dnn.blobFromImage(image=image, scalefactor=0.01, size=(224, 224), mean=(104, 117, 123))
    # Set the input blob for the neural network.
    model.setInput(blob)
    # Forward pass image blog through the model.
    outputs = model.forward()
    
    final_outputs = outputs[0]
    # Make all the outputs 1D.
    final_outputs = final_outputs.reshape(1000, 1)
    # Get the class label.
    label_id = np.argmax(final_outputs)
    # Convert score to probabilities
    probs = np.exp(final_outputs) / np.sum(np.exp(final_outputs))
    # Get the final highest probability
    final_prob = np.max(probs) * 100
    # Map the max confidence to the class label names.
    out_name = class_names[label_id]
    out_text = f"{out_name}, {final_prob:.3f}%"
    return out_text

images = []
imageclasses = []
for img_path in glob.glob('input/*.jpg'):
    img = cv2.imread(img_path)
    images.append(img)
    print("Classifying "+img_path)
    imageclasses.append(classify(img))


for i, image in enumerate(images):
    cv2.putText(image, str(imageclasses[i]), (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
    cv2.imshow(str(imageclasses[i]), image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()