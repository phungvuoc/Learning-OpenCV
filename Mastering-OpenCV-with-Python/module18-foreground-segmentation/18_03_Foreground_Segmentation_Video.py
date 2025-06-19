import cv2
import mediapipe as mp
import numpy as np
  
# Initializing mediapipe segmentation class.
mp_selfie_segmentation = mp.solutions.selfie_segmentation
# Setup segmentation function.
segment = mp_selfie_segmentation.SelfieSegmentation()


# No Blur.
def noBlur(img, bg_img, threshold = 0.3, mode = 'replace'):
  img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

  results = segment.process(img)

  binary_mask = results.segmentation_mask > threshold
  # Convert the mask to a three channel image.
  mask = np.dstack((binary_mask, binary_mask, binary_mask))

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if mode == 'replace' :
    output_image = np.where(mask, img, bg_img)

  elif mode == 'deSaturatedOriginal':
    bg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Convert the gray background image to a three channel image.
    bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
    output_image = np.where(mask, img, bg_img_gray)

  elif mode == 'deSturatedReplaced':
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    # Convert the gray background image to a three channel image.
    bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
    output_image = np.where(mask, img, bg_img_gray)

  else:
    print('Invalid mode provided.')

  return output_image

# Blur.
def withBlur(img, bg_img, ksize = 25, threshold = 0.3, mode = 'original'):
  img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

  results = segment.process(img)

  binary_mask = results.segmentation_mask > threshold
  # Convert the mask to a three channel image.
  mask = np.dstack((binary_mask, binary_mask, binary_mask))

  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

  if mode == 'original':
    # Create a blurred copy of the input image.
    blurred_image = cv2.GaussianBlur(img, (ksize, ksize), 0)
    output_image = np.where(mask, img, blurred_image)

  elif mode == 'replace':
    # Create a blurred copy of the input image.
    blurred_image = cv2.GaussianBlur(bg_img, (ksize, ksize), 0)
    output_image = np.where(mask, img, blurred_image)

  elif mode == 'deSaturatedOriginal':
    # Create a blurred copy of the input image.
    blurred_image = cv2.GaussianBlur(img, (ksize, ksize), 0)
    desat_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    desat_blurred = cv2.cvtColor(desat_blurred, cv2.COLOR_GRAY2BGR)
    output_image = np.where(mask, img, desat_blurred)

  elif mode == 'deSturatedReplaced':
    # Create a blurred copy of the input image.
    blurred_image = cv2.GaussianBlur(bg_img, (ksize, ksize), 0)
    desat_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
    desat_blurred = cv2.cvtColor(desat_blurred, cv2.COLOR_GRAY2BGR)
    output_image = np.where(mask, img, desat_blurred)

  else:
    print('Invalid mode provided.')

  return output_image


if __name__ == "__main__":
  # Load background image.
  bg_image = cv2.imread('parthenon.jpg')

  # Create video capture object.
  cap = cv2.VideoCapture(0)

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Can not access camera")
      break
    bg_image = cv2.resize(bg_image, (image.shape[1], image.shape[0]))
    output = withBlur(image, bg_image, ksize = 25, threshold = 0.3, mode = 'deSturatedReplaced')
    cv2.imshow('Output', output)
    k = cv2.waitKey(1)
    if k == 113:
      break

  cap.release()
  cv2.destroyAllWindows()