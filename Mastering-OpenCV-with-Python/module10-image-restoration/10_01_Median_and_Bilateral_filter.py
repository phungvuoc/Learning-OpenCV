import cv2
import matplotlib.pyplot as plt

# Load an images with Salt and pepper noise.
img1 = cv2.imread('images/mona_lisa.jpg')
img2 = cv2.imread('images/ice-flakes-microscopy-salt-and-pepper-noise.jpg')

# Apply median filter.
img1_median = cv2.medianBlur(img1, 9)
img2_median = cv2.medianBlur(img2, 3)

# Apply Gaussian filter for comparison.
img1_gaussian = cv2.GaussianBlur(img1, (5, 5), cv2.BORDER_DEFAULT)
img2_gaussian = cv2.GaussianBlur(img2, (5, 5), cv2.BORDER_DEFAULT)

# Load an images with Salt and pepper noise.
img1 = cv2.imread('images/mona_lisa.jpg')
img2 = cv2.imread('images/ice-flakes-microscopy-salt-and-pepper-noise.jpg')

# Apply median filter.
img1_median = cv2.medianBlur(img1, 9)
img2_median = cv2.medianBlur(img2, 3)

# Apply Gaussian filter for comparison.
img1_gaussian = cv2.GaussianBlur(img1, (5, 5), cv2.BORDER_DEFAULT)
img2_gaussian = cv2.GaussianBlur(img2, (5, 5), cv2.BORDER_DEFAULT)

temp1 = cv2.hconcat([img1, img1_gaussian, img1_median])
temp2 = cv2.hconcat([img2, img2_gaussian, img2_median])

cv2.imshow('Original :: Gaussina Blur :: Median filter', temp1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Original :: Gaussina Blur :: Median filter', temp2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load image with gaussian noise.
image1 = cv2.imread('images/mri-skull-20-percent-gaussian-noise.jpg')
image2 = cv2.imread('images/mri-skull-40-percent-gaussian-noise.jpg')

# diameter of the pixel neighborhood used during filtering.
dia = 20

# Larger the value the distant colours will be mixed together
# to produce areas of semi equal colors.
sigmaColor = 200

# Larger the value more the influence of the farther placed pixels 
# as long as their colors are close enough.
sigmaSpace = 100

# Apply bilateralFilter.
dst1 = cv2.bilateralFilter(image1, dia, sigmaColor, sigmaSpace)
dst2 = cv2.bilateralFilter(image2, dia, sigmaColor, sigmaSpace)

# Resizing for display convenience, not necessary otherwise.
image1_res = cv2.resize(image1, None, fx = 0.3, fy = 0.3)
image2_res = cv2.resize(image2, None, fx = 0.3, fy = 0.3)
dst1_res = cv2.resize(dst1, None, fx = 0.3, fy = 0.3)
dst2_res = cv2.resize(dst2, None, fx = 0.3, fy = 0.3)

temp1 = cv2.hconcat([image1_res, dst1_res])
temp2 = cv2.hconcat([image2_res, dst2_res])
temp3 = cv2.vconcat([temp1, temp2])

cv2.imshow('Image with Gaussian noise :: Image after Bilateral filter', temp3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load images. 
img1 = cv2.imread('images/face-original.jpg')
img2 = cv2.imread('images/girl-skin.jpg')

# Apply Gaussian filter for comparison.
img1_gaussian = cv2.GaussianBlur(img1, (5,5), cv2.BORDER_DEFAULT)
img2_gaussian = cv2.GaussianBlur(img2, (5,5), cv2.BORDER_DEFAULT)

# Apply bilateralFilter.
img1_bilateral = cv2.bilateralFilter(img1, d = 25, sigmaColor = 90, sigmaSpace = 40)
img2_bilateral = cv2.bilateralFilter(img2, d = 30, sigmaColor = 65, sigmaSpace = 15)

# Resizing for display convenience, not necessary otherwise.
img1_res = cv2.resize(img1, None, fx = 0.7, fy = 0.7)
img2_res = cv2.resize(img2, None, fx = 0.3, fy = 0.3)
img1_gaussian_res  = cv2.resize(img1_gaussian, None,  fx = 0.7, fy = 0.7)
img2_gaussian_res  = cv2.resize(img2_gaussian, None,  fx = 0.3, fy = 0.3)
img1_bilateral_res = cv2.resize(img1_bilateral, None, fx = 0.7, fy = 0.7)
img2_bilateral_res = cv2.resize(img2_bilateral, None, fx = 0.3, fy = 0.3)

temp1 = cv2.vconcat([img1_res, img1_gaussian_res, img1_bilateral_res])
temp2 = cv2.vconcat([img2_res, img2_gaussian_res, img2_bilateral_res])

cv2.imshow('Original :: Gaussian Blur applied :: Bilateral filter applied', temp1)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Original :: Gaussian Blur applied :: Bilateral filter applied', temp2)
cv2.waitKey(0)
cv2.destroyAllWindows()

