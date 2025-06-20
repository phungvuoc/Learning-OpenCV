import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
plt.rcParams['image.cmap'] = 'gray'


# Function to align bounding boxes.
def fourPointsTransform(frame, vertices):
	"""Extracts and transforms roi of frame defined by vertices into a rectangle."""
	# Get vertices of each bounding box 
	vertices = np.asarray(vertices).astype(np.float32)
	outputSize = (100, 32)
	targetVertices = np.array([
		[0, outputSize[1] - 1],
		[0, 0],
		[outputSize[0] - 1, 0],
		[outputSize[0] - 1, outputSize[1] - 1]], dtype="float32")
	# Apply perspective transform
	rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
	result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
	return result


def recognizeText(image, debug=False):
	"""Detect, recognize, and output text from a natural image scene."""
	# Create a blank matrix to be used to display the output image
	outputCanvas = np.full(image.shape[:3], 255, dtype=np.uint8)

	# Use the DB text detector initialised previously to detect the presence of text in the image
	boxes, confs = textDetector.detect(image)

	print("Recognized Text:")
	# Iterate throught the bounding boxes detected by the text detector model
	for box in boxes:
		# Apply transformation on the bounding box detected by the text detection algorithm.
		croppedRoi = fourPointsTransform(image, box)
		if debug:
			cv2.imshow('Output', croppedRoi)

		# Recognise the text using the crnn model.
		recResult = textRecognizer.recognize(croppedRoi)
		print(recResult)

		# Get scaled values.
		boxHeight = int((abs((box[0, 1] - box[1, 1]))))

		# Get scale of the font.
		fontScale = cv2.getFontScaleFromHeight(
			cv2.FONT_HERSHEY_SIMPLEX, boxHeight-10, 1)

		# Write the recognized text on the output image.
		placement = (int(box[0,0]), int(box[0,1]))
		cv2.putText(outputCanvas, recResult, placement,
			cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 1, 5)

	# Draw the bounding boxes of text detected.
	cv2.polylines(image, boxes, True, (255, 0, 255), 4)

	# Concatenate the input image with the output image.
	combinedResult = cv2.hconcat([image, outputCanvas])
	cv2.imshow('Result', combinedResult)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":

	# Load Image
	image = cv2.imread('./visuals/dutch_signboard.jpg')
	cv2.imshow('Image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Define list to store the vocabulary, the recognizable characters.
	vocabulary =[]

	# Open file to import the vocabulary.
	with open("./resources/alphabet_94.txt") as f:
	    # Read the file line by line, and append each into the vocabulary list.
	    for l in f:
	        vocabulary.append(l.strip())
	    f.close()
	print("Vocabulary:", vocabulary)
	print("Vocabulary size: ", len(vocabulary))

	# DB model for text-detection based on resnet50.
	textDetector = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")

	inputSize = (640, 640)

	# Set threshold for Binary Map creation and polygon detection.
	binThresh = 0.3
	polyThresh = 0.5

	mean = (122.67891434, 116.66876762, 104.00698793)

	textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
	textDetector.setInputParams(1.0/255, inputSize, mean, True)

	# Import the pre-built Recognition model files.
	# CRNN model for text-recognition.
	textRecognizer = cv2.dnn_TextRecognitionModel("./resources/crnn_cs.onnx")
	textRecognizer.setDecodeType("CTC-greedy")
	textRecognizer.setVocabulary(vocabulary)
	textRecognizer.setInputParams(1/127.5, (100,32), (127.5, 127.5, 127.5), True)

	# Use the DB text detector initialized previously to detect the presence of text in the image.
	boxes, confs = textDetector.detect(image)

	# Draw the bounding boxes of text detected.
	cv2.polylines(image, boxes, True, (255, 0, 255), 4)
	# Display the image with the bounding boxes drawn
	cv2.imshow('Bounding Boxes', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Display the transformed output of the first detected text box.
	warped_detection = fourPointsTransform(image, boxes[0])
	cv2.imshow('Transformed Detected Text', warped_detection)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Recognize text from each box and display on a canvas.
	textData=[]
	outputCanvas = np.full(image.shape[:3], 255, dtype=np.uint8)

	# Iterate throught the bounding boxes detected by the text detector model
	print("Recognized Text:")
	for box in boxes:
		# Apply transformation on the bounding box detected by the text detection algorithm
		croppedRoi  = fourPointsTransform(image,box)

		# Recognise the text using the crnn model
		recResult = textRecognizer.recognize(croppedRoi)

		# Get scaled values
		boxHeight = int((abs((box[0, 1] - box[1, 1]))))

		# Get scale of the font
		fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight-30, 1 )

		# Write the recognized text on the output image
		placement = (int(box[0, 0]), int(box[0, 1]))
		cv2.putText(outputCanvas, recResult, placement,
		            cv2.FONT_HERSHEY_SIMPLEX, fontScale, (255, 0, 0), 1, 5)
		# Append recognized text to the data storage variable
		textData.append(recResult)

	# Join the text data together to form a output sentence
	textData = ' '.join(textData)
	print(textData);
	cv2.imshow('Output Canvas', outputCanvas)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Concatenate input and output images and display the output together.
	# Concatenate the input image with the output image
	combinedResult = cv2.hconcat([image, outputCanvas])
	cv2.imwrite('visuals/english_signboard_recognised.jpg', combinedResult)
	cv2.imshow('Combined Canvas', combinedResult)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	img2 = cv2.imread('./visuals/card.jpg')
	recognizeText(img2)

	img3 = cv2.imread('./visuals/traffic2.jpg')
	recognizeText(img3)

	img4 = cv2.imread('./visuals/car.jpg')
	recognizeText(img4)

	img5 = cv2.imread('./visuals/traffic1.jpg')
	recognizeText(img5)

	# Failure Case.
	img6 = cv2.imread('./visuals/Board.jpg')
	recognizeText(img6)








