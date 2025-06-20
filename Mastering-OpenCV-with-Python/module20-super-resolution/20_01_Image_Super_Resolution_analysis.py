import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 18


# Define a super resolution function.
def super_resolve_OpenCV(image, method, scale):

	# The model name consists of the method and the upsample scale.
	modelPath = './models/{}_x{}.pb'.format(method, scale)
	sr.readModel(modelPath)

	# Set the model by passing the method and the upsampling scale factor.
	sr.setModel(method.lower(), scale) 

	# Upscale the input image.
	result = sr.upsample(image) 

	return result


# Define a function for Comparing Results.
def compare_results(low_res, high_res, scale, roi, u_roi):

	# Generate upsampled results.
	result_bicubic = cv2.resize(low_res, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
	result_EDSR = super_resolve_OpenCV(low_res, 'EDSR', scale)

	# Array slice for cropped ROI.
	roi_y = slice(roi[0],roi[1],1)
	roi_x = slice(roi[2],roi[3],1)

	# Array slice for upsampled ROI.
	u_roi_y = slice(u_roi[0],u_roi[1],1)
	u_roi_x = slice(u_roi[2],u_roi[3],1)

	img_low     = low_res       [roi_y, roi_x]    [:,:,::-1]
	img_bicubic = result_bicubic[u_roi_y, u_roi_x][:,:,::-1]
	img_edsr    = result_EDSR   [u_roi_y, u_roi_x][:,:,::-1]
	img_truth   = high_res      [u_roi_y, u_roi_x][:,:,::-1]

	plt.figure(figsize=[10,10])
	plt.subplot(221); plt.imshow(img_low);     plt.title('Cropped (Low Res) Input')
	plt.subplot(222); plt.imshow(img_bicubic); plt.title('Bicubic (Upsampled)')
	plt.subplot(223); plt.imshow(img_edsr);    plt.title('EDSR (Upsampled)');
	plt.subplot(224); plt.imshow(img_truth);   plt.title('Ground Truth (at upsampled resolution)')
	plt.show()


#================================INITILAIZATIONS==================================================#
# Algorithm Runtime Comparison.
# Initialize accumulators.
totalBicubic = 0
totalEDSR    = 0
totalESPCN   = 0
totalLAPSRN  = 0
totalFSRCNN  = 0

# Set Scale for upsampling.
scale = 4

# Number of iterations for time comparison.
iterations = 10

# Create a super res object. 
sr = cv2.dnn_superres.DnnSuperResImpl_create()
#==================================================================================================#


if __name__ == "__main__":
	# Test image.
	# Read a hi-res version of the logo.
	logo_high = cv2.imread('opencv-logo.png')
	# Downsize the hi-res logo.
	logo_low = cv2.resize(logo_high, (64,64), interpolation=cv2.INTER_AREA)

	# Read and display the original and cropped image.
	bike200 = cv2.imread('bike-200.png')
	bike800 = cv2.imread('bike-800.png')
	plt.figure(figsize=[20,15])
	plt.subplot(1,2,1); plt.imshow(bike800[:,:,::-1]); plt.title('Ground Truth (high resolution)')
	plt.subplot(1,2,2); plt.imshow(bike200[:,:,::-1]); plt.title('Low Resolution (4x downsampled)')
	plt.show()

	# Perform Upscaling and Comparison.
	roi = np.array((60,95,70,110))
	upsampled_roi = scale*roi
	compare_results(bike200, bike800, scale, roi, upsampled_roi)

	# Read and display the original and cropped image.
	wasp320 = cv2.imread('wasp-320.jpg')
	wasp1280 = cv2.imread('wasp-1280.jpg')
	plt.figure(figsize=[20,15])
	plt.subplot(1,2,1); plt.imshow(wasp1280[:,:,::-1]); plt.title('Ground Truth (high resolution)')
	plt.subplot(1,2,2); plt.imshow(wasp320 [:,:,::-1]); plt.title('Low Resolution (4x downsampled)')
	plt.show()

	# Perform Upscaling and comparison.
	roi = np.array((100,130,200,240))
	upsampled_roi = scale*roi
	compare_results(wasp320, wasp1280, scale, roi, upsampled_roi)

	# Execute each algorithm using the function defined above and keep adding the time to the variable.
	for i in range(iterations):
	    
	    timeBicubic = time.time()
	    resultBicubic = cv2.resize(logo_low, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
	    totalBicubic += time.time() - timeBicubic    
	    
	    timeFSRCNN = time.time()
	    resultFSRCNN = super_resolve_OpenCV(logo_low, 'FSRCNN', scale)
	    totalFSRCNN += time.time() - timeFSRCNN
	    
	    timeESPCN = time.time()
	    resultESPCN = super_resolve_OpenCV(logo_low, 'ESPCN', scale)
	    totalESPCN += time.time() - timeESPCN
	    
	    timeLAPSRN = time.time()
	    resultLapSRN = super_resolve_OpenCV(logo_low, 'LapSRN', scale)
	    totalLAPSRN += time.time() - timeLAPSRN
	    
	    timeEDSR = time.time()
	    resultEDSR = super_resolve_OpenCV(logo_low, 'EDSR', scale)
	    totalEDSR += time.time() - timeEDSR    

	# Find average time taken for each algorithm.
	avgBicubic = totalBicubic / iterations
	avgEDSR = totalEDSR / iterations
	avgESPCN = totalESPCN / iterations
	avgLAPSRN = totalLAPSRN / iterations
	avgFSRCNN = totalFSRCNN / iterations

	# Plot them on a bar chart.
	plt.figure(figsize=[20,10])

	plt.xticks(fontsize=20)
	plt.yticks(fontsize=12)

	x_axis = ['Bicubic', 'FSRCNN', 'ESPCN', 'LAPSRN', 'EDSR'] 
	y_axis = [avgBicubic, avgFSRCNN, avgESPCN, avgLAPSRN, avgEDSR]
	plt.bar(x_axis, y_axis, color = ['k', 'r', 'g', 'b', 'c'], align='center')

	# Plot the values on top of the bar.
	for x, y, p in zip(range(5), y_axis, y_axis):
	    plt.text(x - .2, y + .01, '{:.4f} sec'.format(p), fontsize = 18)
	    
	plt.ylabel('Execution Time [sec]', fontsize = 18); plt.ylim(0,1); plt.show();

	img_kitten = cv2.imread('kitten-200.png')

	# Upscale the input image.
	kitten_bicubic = cv2.resize(img_kitten, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

	# Read the model.
	sr.readModel('./models/EDSR_x4.pb')

	# Set the model by passing the method and the upsampling scale factor.
	sr.setModel('edsr', 4) 

	# Upscale the input image.
	kitten_edsr= sr.upsample(img_kitten)

	# Display.
	cv2.imshow('Kitten Bicubic', kitten_bicubic)
	cv2.imshow('Kitten EDSR', kitten_edsr)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
