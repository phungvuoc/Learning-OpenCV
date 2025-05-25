import cv2
import numpy as np

# Convolution example.
image_8x8 = np.ones((8, 8), dtype = np.uint8)*90
image_8x8[1:7,1:4] = 20
image_8x8[1:7,4:7] = 150

print(image_8x8)
# Define the Sobel-X kernel.
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]) 

# Convolve the image with Sobel-X 3x3 kernel.
sobelx_filter2d = cv2.filter2D(src = image_8x8, ddepth = cv2.CV_64F, kernel = kernel, borderType = cv2.BORDER_REPLICATE)

# Print the filtered results (intensity gradients)
print('')
print(sobelx_filter2d)
print('')

# Example: Map gradients to [0, 255]
sobelx_filter2d = sobelx_filter2d - sobelx_filter2d.min()
sobelx_filter2d = sobelx_filter2d/sobelx_filter2d.max()
sobelx_filter2d = (sobelx_filter2d * 255).astype('uint8')
print(sobelx_filter2d)

# Resizing for display convenience, not mendatory otherwise.
image_8x8 = cv2.resize(image_8x8, None, fx = 50, fy = 50, interpolation = cv2.INTER_AREA)
sobelx_filter2d = cv2.resize(sobelx_filter2d, None, fx = 50, fy = 50, interpolation = cv2.INTER_AREA)

# Display.
cv2.imshow('Original Image', image_8x8)
cv2.waitKey(0)
cv2.imshow('Intensity Gradient', sobelx_filter2d)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Detect Vertical Edges using Sobel-X Kernel and filter2D().
# Read image.
img = cv2.imread('checkerboard_color.png')
# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print(img.shape)

# Display.
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)

# Define a Kernel and Apply Convolution.
# Define a Sobel-X kernel.
kernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

sobelx = cv2.filter2D(src = img_gray, ddepth = cv2.CV_64F, kernel = kernel)

# Display.
cv2.imshow('Sobelx Edge Map', sobelx)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Using Sobel() to Detect Vertical and Horizontal Edges.
sobelx  = cv2.Sobel(src = img_gray, ddepth = cv2.CV_64F, dx = 1, dy = 0, ksize = 3) 
sobely  = cv2.Sobel(src = img_gray, ddepth = cv2.CV_64F, dx = 0, dy = 1, ksize = 3)

# Display.
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)
cv2.imshow('Sobelx Edge Map', sobelx)
cv2.waitKey(0)
cv2.imshow('Sobely Edge Map', sobely)
cv2.waitKey(0)
cv2.destroyAllWindows()

#  Canny Edge Detection (simple example with no texture or noise).
img = cv2.imread('coca-cola-logo.png')
# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(img_gray, threshold1 = 180, threshold2 = 200)

# Display.
cv2.imshow('Original', img)
cv2.waitKey(0)
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)
cv2.imshow('Canny Edge Map', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Effect of threshold 2.
img = cv2.imread('phone_ipad.jpg')
# Convert to grayscale.
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges1 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 300)
edges2 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 500)
edges3 = cv2.Canny(img_gray, threshold1 = 200, threshold2 = 1000)

# Display.
cv2.imshow('Grayscale', img_gray)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 300', edges1)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 500', edges1)
cv2.waitKey(0)
cv2.imshow('Edges with T2 = 1000', edges1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Canny edge detection with and without blurring.
# Read image.
img1 = cv2.imread('butterfly.jpg')
img2 = cv2.imread('Large_Scaled_Forest_Lizard.jpg')

# Resize for display convenience, not mendatory.
img1 = cv2.resize(img1, None, fx = 0.6, fy = 0.6)
img2 = cv2.resize(img2, None, fx = 0.6, fy = 0.6)
# Convert to gray scale.
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Canny Edge detection without blurring.
original_edges_1 = cv2.Canny(img1_gray, threshold1 = 180, threshold2 = 200)
original_edges_2 = cv2.Canny(img2_gray, threshold1 = 180, threshold2 = 200)

# Apply Gaussian blur with kernel size 7x7.
img1_blur = cv2.GaussianBlur(img1_gray, (7,7), 0)
# Apply Gaussian blur with kernel size 7x7 as the noise is more.
img2_blur = cv2.GaussianBlur(img2_gray, (7,7), 0)

blurred_edges_1 = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
blurred_edges_2 = cv2.Canny(img2_blur, threshold1 = 180, threshold2 = 200)

compare1 = cv2.hconcat([img1_gray, original_edges_1, blurred_edges_1])
compare2 = cv2.hconcat([img2_gray, original_edges_2, blurred_edges_2])

# Display.
cv2.imshow('Original Gray Scale :: Canny Edge without Blurring :: Canny Edge with Blurring', compare1)
cv2.waitKey(0)
cv2.imshow('Original Gray Scale :: Canny Edge without Blurring :: Canny Edge with Blurring', compare2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Hysteresis Thresholding Example (effect of Threshold1).
# Edge detection with a high Threshold1 value.
blurred_edges_tight = cv2.Canny(img1_blur, threshold1 = 180, threshold2 = 200)
# Edge detection with a low Threshold1 value.
blurred_edges_open  = cv2.Canny(img1_blur, threshold1 = 50, threshold2 = 200)

# Display.
cv2.imshow('Threshold1 = 180, Threshold2 = 200', blurred_edges_tight)
cv2.waitKey(0)
cv2.imshow('Threshold1 = 50, Threshold2 = 200', blurred_edges_open)
cv2.waitKey(0)
cv2.destroyAllWindows()