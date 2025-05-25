import cv2
import numpy as np

kernel_size = 5
# Create a 5*5 kernel with all elements equal to 1.
kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / kernel_size**2

print(kernel)

# Perform convolution.
filename = "kitten.jpg"
image = cv2.imread(filename)

dst = cv2.filter2D(image, ddepth=-1, kernel=kernel)

# Display.
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.imshow("Convolution result", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Box blur in OpenCV.
# Apply a box filter - kernel size 5.
box_blur1 = cv2.blur(image, (5, 5))

# Apply a box filter - kernel size 11.
box_blur2 = cv2.blur(image, (11, 11))

# Dispaly.
cv2.imshow("Blur 5x5 kernel", box_blur1)
cv2.waitKey(0)
cv2.imshow("Blur 11x11 kernel", box_blur2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian Blur.
# Apply Gaussian blur.
gaussian_blur1 = cv2.GaussianBlur(image, (5, 5), 0, 0)
gaussian_blur2 = cv2.GaussianBlur(image, (11, 11), 0, 0)

# Dispaly.
cv2.imshow("Gaussian Blur 5x5 kernel", gaussian_blur1)
cv2.waitKey(0)
cv2.imshow("Gaussian Blur 11x11 kernel", gaussian_blur2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Compare box and Gaussian Blur.
compare = cv2.hconcat([box_blur2, gaussian_blur2])
cv2.imshow("Box Blur :: Gaussian Blur", compare)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Gaussian blur and effect of sigma.
# Specifying sigmax = 0 and sigmay = 0, will compute a sigma of 2 for a 11x11 kernal
gaussian_blur3 = cv2.GaussianBlur(image, (11, 11), 0, 0)
# Here we are explicity setting the sigma values to be very large.
gaussian_blur4 = cv2.GaussianBlur(image, (11, 11), 5, 5)

compare = cv2.hconcat([gaussian_blur3, gaussian_blur4])
cv2.imshow("Gaussian Blur sigma 0 :: Gaussian Blur sigma 5", compare)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Image sharpening.
saturn = cv2.imread("saturn.jpg")

# Define a sharpening kernel.
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

saturn_sharp = cv2.filter2D(saturn, ddepth=-1, kernel=kernel)
cv2.imshow("Original", saturn)
cv2.waitKey(0)
cv2.imshow("Sharpened", saturn_sharp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Recovering Sharpness from Gaussian Blur.
image = cv2.imread("kitten_zoom.png")

gaussian_blur = cv2.GaussianBlur(image, (11, 11), 0, 0)

# Sharpening kernel.
kernel1 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

# More extreme sharpening kernel.
kernel2 = np.array([[0, -4, 0], [-4, 17, -4], [0, -4, 0]])

# Apply sharpening.
image_sharp1 = cv2.filter2D(gaussian_blur, ddepth=-1, kernel=kernel1)
image_sharp2 = cv2.filter2D(gaussian_blur, ddepth=-1, kernel=kernel2)

# Apply emboss filter to the original image.
kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
y = np.ones(image.shape[:2], np.uint8) * 128
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
embossed_image = cv2.add(cv2.filter2D(gray, -1, kernel_emboss), y)

cv2.imshow("Original", image)
cv2.waitKey(0)
cv2.imshow("Gaussian Blur (11x11)", gaussian_blur)
cv2.waitKey(0)
cv2.imshow("Sharpened", image_sharp1)
cv2.waitKey(0)
cv2.imshow("Sharpened More", image_sharp2)
cv2.waitKey(0)
cv2.imshow("Embossed Image", embossed_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
