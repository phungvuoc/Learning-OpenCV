import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['axes.titlesize'] = 14


def image_seg_enhance(img, bg_img=None, threshold=0.5, mode='gray'):

    # Convert the image to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segment the image.
    results = segment.process(img)

    # Apply the threshold to create a binary map.
    binary_mask = results.segmentation_mask > threshold

    # Convert the mask to a 3-channel image.
    mask = np.dstack((binary_mask, binary_mask, binary_mask))

    # Convert the image back to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if mode == 'gray':
        # Convert the original background to grayscale.
        bg_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        output_image = np.where(mask, img, bg_img_gray)
    elif mode == 'replace' :
        # Replace the background in the original image with the background image.
        output_image = np.where(mask, img, bg_img)
    elif mode == 'replace_gray':
         # Replace the background in the original image with a grayscale version of the background image.
        bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)
        output_image = np.where(mask, img, bg_img_gray)
    else:
        output_image = None
        print('Invalid mode provided.')

    return output_image


def image_seg_enhance_blur(img, bg_img=None, ksize=25, threshold=0.3, mode='blur'):

    # Convert the image to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segment the image.
    results = segment.process(img)

    # Apply the threshold to create a binary map.
    binary_mask = results.segmentation_mask > threshold

    # Convert the mask to a 3-channel image.
    mask = np.dstack((binary_mask, binary_mask, binary_mask))

    # Convert the image back to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if mode == 'blur':
        # Blur the background in the original image.
        blurred_image = cv2.GaussianBlur(img, (ksize, ksize), 0)
        output_image = np.where(mask, img, blurred_image)
    elif mode == 'gray':
        # Convert the original background to garyscale and blur.
        blurred_image = cv2.GaussianBlur(img, (ksize, ksize), 0)
        desat_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        desat_blurred = cv2.cvtColor(desat_blurred, cv2.COLOR_GRAY2BGR)
        output_image = np.where(mask, img, desat_blurred)
    elif mode == 'replace':
        # Replace the background in the original image with a blurred version of the background image.
        blurred_image = cv2.GaussianBlur(bg_img, (ksize, ksize), 0)
        output_image = np.where(mask, img, blurred_image)
    elif mode == 'replace_gray':
        # Replace the background in the original image with a blurred/grayscale version of the background image.
        blurred_image = cv2.GaussianBlur(bg_img, (ksize, ksize), 0)
        desat_blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        desat_blurred = cv2.cvtColor(desat_blurred, cv2.COLOR_GRAY2BGR)
        output_image = np.where(mask, img, desat_blurred)
    else:
        output_image = None
        print('Invalid mode provided.')

    return output_image


if __name__ == "__main__":
    # Read the target (foreground) image.
    img = cv2.imread('girl.png')

    # Read a background image.
    bg_img = cv2.imread('parthenon.jpg')

    # Resize the background image to be the same size as the target image.
    bg_img = cv2.resize(bg_img, (img.shape[1], img.shape[0]))

    cv2.imshow('Foreground Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imshow('Background Image', bg_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Initialize MediaPipe Selfie Segmentation.
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    segment = mp_selfie_segmentation.SelfieSegmentation(model_selection=0)

    # Segment the Foreground.
    # Convert to RGB.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segment the original image.
    results = segment.process(img)

    # Convert to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Retrieve segmentation mask from results.
    img_seg_mask = results.segmentation_mask

    # Apply a threhsold to generate a binary mask.
    threshold = 0.5
    binary_mask = img_seg_mask > threshold

    cv2.imshow('Segmentation Mask', img_seg_mask)
    cv2.waitKey(0)
    cv2.imshow('Binary Mask', np.uint8(255*binary_mask))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Background Replacement.

    # Convert the mask to a 3 channel image.
    mask3d = np.dstack((binary_mask, binary_mask, binary_mask))

    # Apply the mask to the original image and a new backgroud image.
    img_out = np.where(mask3d, img, bg_img)

    cv2.imshow('Final Output', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create a 3-channel grayscale background image.
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)

    # Apply the mask to the original image and grayscale version of the background image.
    img_out = np.where(mask3d, img, bg_img_gray)

    cv2.imshow('Final Output Saturated', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Segmentation Threshold Experiment.
    # Create list of thresholds.
    threshold = [0.2, 0.5, 0.8]

    title       = []
    img_out     = []
    img_out_bw  = []
    mask3d      = []
    binary_mask = []
    # Create plot titles.
    for idx in range(len(threshold)):
        temp = 'Threshold: ' + str(threshold[idx])
        title.append(temp)

    # Convert to RGB for MediaPipe.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Segment image foreground.
    results = segment.process(img)

    # Convert back to BGR.
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Create a 3-channel grayscale background image.
    bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    bg_img_gray = cv2.cvtColor(bg_gray, cv2.COLOR_GRAY2BGR)

    # Process test cases.
    for idx in range(len(threshold)):
        binary_mask.append(results.segmentation_mask > threshold[idx])
        mask3d.append(np.dstack((binary_mask[idx], binary_mask[idx], binary_mask[idx])))
        img_out.append(np.where(mask3d[idx], img, bg_img))
        img_out_bw.append(np.where(mask3d[idx], img, bg_img_gray))

    # Plot test cases.
    for idx in range(len(threshold)):
        plt.figure(figsize = (20,8))
        plt.subplot(141); plt.axis('off'); plt.imshow(img[:,:,::-1]);             plt.title('Original / Target Image')
        plt.subplot(142); plt.axis('off'); plt.imshow(binary_mask[idx]);          plt.title(title[idx])
        plt.subplot(143); plt.axis('off'); plt.imshow(img_out[idx][:,:,::-1]);    plt.title(title[idx])
        plt.subplot(144); plt.axis('off'); plt.imshow(img_out_bw[idx][:,:,::-1]); plt.title(title[idx]);

        file_out    = 'img_out_thresh_' + str(threshold[idx]) + '.png'
        file_out_bw = 'img_out_bw_thresh_' + str(threshold[idx]) + '.png'
        cv2.imwrite(file_out,    img_out[idx])
        cv2.imwrite(file_out_bw, img_out_bw[idx])

    # Application options.
    thresh = 0.6

    # Using: image_seg_enhance()
    bg_gray     = image_seg_enhance(img, threshold = thresh, mode = 'gray')
    bg_rep      = image_seg_enhance(img, bg_img, threshold = thresh, mode = 'replace')
    bg_rep_gray = image_seg_enhance(img, bg_img, threshold = thresh, mode = 'replace_gray')

    # Using: image_seg_enhance_blur()
    bg_blur          = image_seg_enhance_blur(img, ksize = 71, threshold = thresh, mode = 'blur')
    bg_gray_blur     = image_seg_enhance_blur(img, ksize = 71, threshold = thresh, mode = 'gray')
    bg_rep_blur      = image_seg_enhance_blur(img, bg_img, ksize = 21, threshold = thresh, mode = 'replace')
    bg_rep_gray_blur = image_seg_enhance_blur(img, bg_img, ksize = 21, threshold = thresh, mode = 'replace_gray')

    plt.figure(figsize = (18,12))

    plt.subplot(241); plt.axis('off'); plt.imshow(img[...,::-1]);              plt.title('Original / Target Image')
    plt.subplot(242); plt.axis('off'); plt.imshow(bg_blur[...,::-1]);          plt.title('BG Blurred')
    plt.subplot(243); plt.axis('off'); plt.imshow(bg_gray[...,::-1]);          plt.title('BG Gray')
    plt.subplot(244); plt.axis('off'); plt.imshow(bg_gray_blur[...,::-1]);     plt.title('BG Gray Blurred')

    plt.subplot(245); plt.axis('off'); plt.imshow(bg_rep[...,::-1]);           plt.title('BG Replaced')
    plt.subplot(246); plt.axis('off'); plt.imshow(bg_rep_blur[...,::-1]);      plt.title('BG Replaced Blurred')
    plt.subplot(247); plt.axis('off'); plt.imshow(bg_rep_gray[...,::-1]);      plt.title('BG Replaced Gray')
    plt.subplot(248); plt.axis('off'); plt.imshow(bg_rep_gray_blur[...,::-1]); plt.title('BG Replaced Gray Blurred');
    plt.show()