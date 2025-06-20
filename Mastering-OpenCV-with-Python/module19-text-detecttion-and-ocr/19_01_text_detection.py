import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'

def detectTextAll(image):
	"""Performs each type of text detection over an image and displays all results side by side."""
	# Making copies of the original image
	imEAST = image.copy()
	imDB18 = image.copy()
	imDB50 = image.copy()

	# Use each text detector to detect the presence of text in the image
	boxesEAST, confsEAST = textDetectorEAST.detect(image)
	boxesDB18, confsDB18 = textDetectorDB18.detect(image)
	boxesDB50, confsDB50 = textDetectorDB50.detect(image)

	# Draw the bounding boxes onto the respective copies of the original image.
	cv2.polylines(imEAST, boxesEAST, True, (255, 0, 255), 4)
	cv2.polylines(imDB18, boxesDB18, True, (255, 0, 255), 4)
	cv2.polylines(imDB50, boxesDB50, True, (255, 0, 255), 4)

	output = cv2.hconcat([image, imEAST, imDB18, imDB50])
	cv2.imshow('Original | EAST | DB18 | DB50', cv2.resize(output, None, fx=0.4, fy=0.4))
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# Load image.
	image = cv2.imread('./visuals/dutch_signboard.jpg')

	cv2.imshow('Image', image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Set input image size.
	inputSize = (320, 320)

	# Load pre-trained models.
	# East model for text-detection
	textDetectorEAST = cv2.dnn_TextDetectionModel_EAST("./resources/frozen_east_text_detection.pb")

	# Set the Detection Confidence Threshold and NMS threshold
	conf_thresh = 0.8
	nms_thresh = 0.4

	textDetectorEAST.setConfidenceThreshold(conf_thresh).setNMSThreshold(nms_thresh)
	textDetectorEAST.setInputParams(1.0, inputSize, (123.68, 116.78, 103.94), True)

	# DB model for text-detection based on resnet50
	textDetectorDB50 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet50.onnx")
	# DB model for text-detection based on resnet18
	textDetectorDB18 = cv2.dnn_TextDetectionModel_DB("./resources/DB_TD500_resnet18.onnx")

	# Set threshold for Binary Map creation and polygon detection
	bin_thresh = 0.3
	poly_thresh = 0.5

	mean = (122.67891434, 116.66876762, 104.00698793)

	textDetectorDB18.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
	textDetectorDB18.setInputParams(1.0/255, inputSize, mean, True)

	textDetectorDB50.setBinaryThreshold(bin_thresh).setPolygonThreshold(poly_thresh)
	textDetectorDB50.setInputParams(1.0/255, inputSize, mean, True)

	# Making copies of the original image
	imEAST = image.copy()
	imDB18 = image.copy()
	imDB50 = image.copy()

	# Use the East text detector to detect the presence of text in the image
	boxesEAST, confsEAST = textDetectorEAST.detect(image)

	# Use the DB18 text detector to detect the presence of text in the image
	boxesDB18, confsDB18 = textDetectorDB18.detect(image)

	# Use the DB50 text detector to detect the presence of text in the image
	boxesDB50, confsDB50 = textDetectorDB50.detect(image)

	# Inspect the output of one of the detected text boxes
	print(boxesEAST[0])

	# Draw the bounding boxes of text detected using EAST.
	cv2.polylines(imEAST, boxesEAST, isClosed=True, color=(255, 0, 255), thickness=4)
	cv2.imshow('Bounding boxes for EAST', imEAST)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Draw the bounding boxes of text detected using DB18.
	cv2.polylines(imDB18, boxesDB18, True, (255, 0, 255), 4)
	cv2.imshow('Bounding boxes for DB18', imDB18)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Draw the bounding boxes of text detected using DB50.
	cv2.polylines(imDB50, boxesDB50, True, (255, 0, 255), 4)
	cv2.imshow('Bounding boxes for DB50', imDB50)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	output = cv2.hconcat([image, imEAST, imDB18, imDB50])
	cv2.imwrite('./visuals/english_signboard_detected.jpg', output)
	cv2.imshow('Original | EAST | DB18 | DB50', cv2.resize(output, None, fx=0.6, fy=0.6))
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	# Other examples.
	img1 = cv2.imread('./visuals/traffic1.jpg')
	detectTextAll(img1)

	img2 = cv2.imread('./visuals/traffic2.jpg')
	detectTextAll(img2)

	img3 = cv2.imread('./visuals/card.jpg')
	detectTextAll(img3)

	img4 = cv2.imread('./visuals/Board.jpg')
	detectTextAll(img4)

	img5 = cv2.imread('./visuals/paper.jpg')
	detectTextAll(img5)

	img6 = cv2.imread('./visuals/car.jpg')
	detectTextAll(img6)

	# Runtime Comparison for EAST vs DB.
	totalEAST = 0
	totalDB18 = 0
	totalDB50 = 0

	iterations = 10
	for i in range(iterations):
	    timeEAST = time.time()
	    result = textDetectorEAST.detect(img4)
	    totalEAST += time.time() - timeEAST
	    
	    timeDB18 = time.time()
	    result = textDetectorDB18.detect(img4)
	    totalDB18 += time.time() - timeDB18
	    
	    timeDB50 = time.time()
	    result = textDetectorDB50.detect(img4)
	    totalDB50 += time.time() - timeDB50

	avgEAST = totalEAST / iterations
	avgDB18 = totalDB18 / iterations
	avgDB50 = totalDB50 / iterations

	plt.bar(['EAST', 'DB18', 'DB50'], [avgEAST, avgDB18, avgDB50], color = ['g', 'b', 'c'])
	plt.ylabel("seconds")
	plt.xlabel("Model")
	plt.show()







