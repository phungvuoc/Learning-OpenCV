import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18

# Load high resolution image.
logo_high = cv2.imread('opencv-logo.png')
   
# Downsize the hi-res logo.
logo_low = cv2.resize(logo_high, (64,64), interpolation=cv2.INTER_AREA)
# Display.
cv2.imshow('High Resolution Image', logo_high)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('Down sampled Image', logo_low)
cv2.waitKey(0)

# Upsample the test images using bi-cubic interpolation.
result_bicubic = cv2.resize(logo_low, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

# Specify the ROI for the 'O'.
roi_y = slice(194,242,1)
roi_x = slice(27,75,1)

# Display.
cv2.imshow('Upsampled Bicubic', result_bicubic)
cv2.imshow('Cropped Patch', result_bicubic[roi_y,roi_x])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Create a super resolution object.
sr = cv2.dnn_superres.DnnSuperResImpl_create()

# Read the model.
sr.readModel('./models/FSRCNN_x4.pb')

# Set the model by passing the method and the upsampling scale factor.
sr.setModel('fsrcnn', 4) 

# Upscale the input image.
result_FSRCNN = sr.upsample(logo_low)

# Display.
cv2.imshow('Upsampled FSRCNN', result_FSRCNN)
cv2.imshow('Upsampled Cropped Patch', result_FSRCNN[roi_y,roi_x])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Super Resolution using ESPCN.
# Read the model.
sr.readModel('./models/ESPCN_x4.pb')

# Set the model by passing the method and the upsampling scale factor.
sr.setModel('espcn', 4) 

# Upscale the input image.
result_ESPCN = sr.upsample(logo_low)

# Display.
cv2.imshow('Upsampled FSRCNN', result_ESPCN)
cv2.imshow('Upsampled Cropped Patch', result_ESPCN[roi_y,roi_x])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Super Resolution using LapSRN.
# Read the model.
sr.readModel('./models/LapSRN_x4.pb')

# Set the model by passing the method and the upsampling scale factor.
sr.setModel('lapsrn', 4) 

# Upscale the input image.
result_LapSRN = sr.upsample(logo_low)

# Display.
cv2.imshow('Upsampled FSRCNN', result_LapSRN)
cv2.imshow('Upsampled Cropped Patch', result_LapSRN[roi_y,roi_x])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Super Resolution Using EDSR.
# Read the model.
sr.readModel('./models/EDSR_x4.pb')

# Set the model by passing the method and the upsampling scale factor.
sr.setModel('edsr', 4) 

# Upscale the input image.
result_EDSR = sr.upsample(logo_low)

# Display.
cv2.imshow('Upsampled FSRCNN', result_EDSR)
cv2.imshow('Upsampled Cropped Patch', result_EDSR[roi_y,roi_x])
cv2.waitKey(0)
cv2.destroyAllWindows()
print(logo_low.shape)
print(result_bicubic[roi_y,roi_x].shape)
print(result_FSRCNN[roi_y,roi_x].shape)

# Compare.
plt.figure(figsize=[20,15])
plt.subplot(231); plt.imshow(logo_low[49:60,7:18][:,:,::-1]);        plt.title('Low Res')
plt.subplot(232); plt.imshow(result_bicubic[roi_y,roi_x][:,:,::-1]); plt.title('Bicubic')
plt.subplot(233); plt.imshow(result_FSRCNN [roi_y,roi_x][:,:,::-1]); plt.title('FSRCNN')
plt.subplot(234); plt.imshow(result_ESPCN  [roi_y,roi_x][:,:,::-1]); plt.title('ESPCN')
plt.subplot(235); plt.imshow(result_LapSRN [roi_y,roi_x][:,:,::-1]); plt.title('LapSRN')
plt.subplot(236); plt.imshow(result_EDSR   [roi_y,roi_x][:,:,::-1]); plt.title('EDSR');
plt.show()

# Compare results.
plt.figure(figsize=[20,15])
plt.subplot(231); plt.imshow(logo_low[:,:,::-1]);       plt.title('Low Res')
plt.subplot(232); plt.imshow(result_bicubic[:,:,::-1]); plt.title('Bicubic')
plt.subplot(233); plt.imshow(result_FSRCNN [:,:,::-1]); plt.title('FSRCNN')
plt.subplot(234); plt.imshow(result_ESPCN  [:,:,::-1]); plt.title('ESPCN')
plt.subplot(235); plt.imshow(result_LapSRN [:,:,::-1]); plt.title('LapSRN')
plt.subplot(236); plt.imshow(result_EDSR   [:,:,::-1]); plt.title('EDSR');
plt.show()