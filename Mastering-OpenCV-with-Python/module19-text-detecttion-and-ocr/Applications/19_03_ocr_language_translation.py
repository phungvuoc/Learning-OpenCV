import cv2
import numpy as np
import pyttsx3
import googletrans
from matplotlib import pyplot as plt


# Align text boxes.
# This Function does transformation over the bounding boxes detected by the text detection model
def fourPointsTransform(frame, vertices):
    # Print vertices of each bounding box
    vertices = np.asarray(vertices).astype(np.float32)
    outputSize = (100, 32)
    targetVertices = np.array(
        [[0, outputSize[1] - 1], [0, 0], [outputSize[0] - 1, 0], [outputSize[0] - 1, outputSize[1] - 1]],
        dtype="float32",
    )
    # Apply perspective transform
    rotationMatrix = cv2.getPerspectiveTransform(vertices, targetVertices)
    result = cv2.warpPerspective(frame, rotationMatrix, outputSize)
    return result


# Perform Language Translation on Recognized Text.
def recognizeTranslateText(image, dest="en", src="", debug=False):

    # Create a blank matrix to be used to display the output image
    outputCanvas = np.full(image.shape[:3], 255, dtype=np.uint8)

    # Use the DB text detector initialised previously to detect the presence of text in the image
    boxes, confs = textDetector.detect(image)

    # Iterate throught the bounding boxes detected by the text detector model
    for box in boxes:

        # Apply transformation on the bounding box detected by the text detection algorithm
        croppedRoi = fourPointsTransform(image, box)
        if debug:
            plt.imshow(croppedRoi)
            plt.show()

        # Recognise the text using the crnn model
        recognizedText = textRecognizer.recognize(croppedRoi)
        if src:
            translation = translator.translate(recognizedText, dest, src)
        else:
            translation = translator.translate(recognizedText, dest)
        print(
            "Recognized Text[{}]: {} -> Translated Text[{}]: {}".format(
                googletrans.LANGUAGES[translation.src], recognizedText, googletrans.LANGUAGES[dest], translation.text
            )
        )

        # Get scaled values
        boxHeight = int((abs((box[0, 1] - box[1, 1]))))

        # Get scale of the font
        fontScale = cv2.getFontScaleFromHeight(cv2.FONT_HERSHEY_SIMPLEX, boxHeight - 5, 1)

        # Write the recognised text on the output image
        cv2.putText(
            outputCanvas,
            translation.text,
            (int(box[0, 0]), int(box[0, 1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale,
            (255, 0, 0),
            1,
            5,
        )

    # Draw the bounding boxes of text detected.
    cv2.polylines(image, boxes, True, (255, 0, 255), 3)

    # Concatenate the input image with the output image
    combinedResult = cv2.hconcat([image, outputCanvas])

    cv2.imshow("Combined Result", combinedResult)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(googletrans.LANGUAGES)

    # Create a Translator Object
    translator = googletrans.Translator()

    # Load Image
    image = cv2.imread("../visuals/dutch_signboard.jpg")
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Import the Language Vocabulary to be used for the text recognition.
    # Define list to store the vocabulary in
    vocabulary = []
    # Open file to import the vocabulary
    with open("../resources/alphabet_94.txt") as f:
        # Read the file line by line
        for l in f:
            # Append each line into the vocabulary list.
            vocabulary.append(l.strip())
        # Close the file
        f.close()
    print("Vocabulary:", vocabulary, len(vocabulary))

    # DB model for text-detection based on resnet50
    textDetector = cv2.dnn_TextDetectionModel_DB("../resources/DB_TD500_resnet50.onnx")

    inputSize = (640, 640)

    # Set threshold for Binary Map creation and polygon detection
    binThresh = 0.3
    polyThresh = 0.5

    mean = (122.67891434, 116.66876762, 104.00698793)

    textDetector.setBinaryThreshold(binThresh).setPolygonThreshold(polyThresh)
    textDetector.setInputParams(1.0 / 255, inputSize, mean, True)

    # CRNN model for text-recognition
    textRecognizer = cv2.dnn_TextRecognitionModel("../resources/crnn_cs.onnx")
    textRecognizer.setDecodeType("CTC-greedy")
    textRecognizer.setVocabulary(vocabulary)
    textRecognizer.setInputParams(1 / 127.5, (100, 32), (127.5, 127.5, 127.5), True)

    # Example 1.
    image = cv2.imread("../visuals/dutch_signboard.jpg")
    recognizeTranslateText(image)

    # Example 2.
    image = cv2.imread("../visuals/german_sign.jpg")
    recognizeTranslateText(image)

    # Example 3.
    image = cv2.imread("../visuals/german_sign.jpg")
    recognizeTranslateText(image, src="de")
