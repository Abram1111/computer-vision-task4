import numpy as np
import cv2
def otsu_threshold(image):
    # Compute histogram of image intensities
    hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))

    # Compute probabilities of each intensity level
    p = hist / float(np.sum(hist))

    # Compute cumulative sums of probabilities and intensity values
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))

    # Compute global mean intensity value
    global_mean = mu[-1]

    # Compute between-class variance for all possible threshold values
    sigma_b_squared = (global_mean * omega - mu) ** 2 / (omega * (1 - omega))

    # Find threshold value that maximizes between-class variance
    max_idx = np.argmax(sigma_b_squared)
    threshold = bins[max_idx]

    # Apply threshold to image
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 1

    return binary_image, threshold

def iterative_thresholding(image, tol=0.5):
    # Initialize threshold value to midpoint of intensity range
    low, high = np.min(image), np.max(image)
    threshold = (low + high) / 2

    # Iterate until threshold value converges
    while True:
        # Divide image into foreground and background regions
        foreground = image > threshold
        background = image <= threshold

        # Compute mean intensities of foreground and background regions
        mean_foreground = np.mean(image[foreground])
        mean_background = np.mean(image[background])

        # Compute new threshold value as average of region mean intensities
        new_threshold = (mean_foreground + mean_background) / 2

        # Check if new threshold value has converged
        if abs(new_threshold - threshold) < tol:
            break

        # Update threshold value and continue iteration
        threshold = new_threshold

    # Apply final threshold value to image
    binary_image = np.zeros_like(image)
    binary_image[image > threshold] = 1

    return binary_image, threshold



def spectral_thresholding(img):
    # Convert the image to grayscale
    if len(img.shape) > 2:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray_img = img

    # Compute the histogram of the image
    hist, bins = np.histogram(gray_img, 256, [0, 256])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()

    # Compute the normalized CDF
    # cdf_normalized = cdf / float(cdf.max())

    # Calculate the mean of the entire image
    mean = np.sum(np.arange(256) * hist) / float(gray_img.size)

    # Initialize variables for the optimal threshold values and the maximum variance
    optimal_high = 0
    optimal_low = 0
    max_variance = 0

    # Loop over all possible threshold values, select ones with maximum variance between modes
    for high in range(0, 256):
        for low in range(0, high):
            w0 = np.sum(hist[0:low])
            if w0 == 0:
                continue
            mean0 = np.sum(np.arange(0, low) * hist[0:low]) / float(w0)

            # Calculate the weight and mean of the low pixels
            w1 = np.sum(hist[low:high])
            if w1 == 0:
                continue
            mean1 = np.sum(np.arange(low, high) * hist[low:high]) / float(w1)

            # Calculate the weight and mean of the high pixels
            w2 = np.sum(hist[high:])
            if w2 == 0:
                continue
            mean2 = np.sum(np.arange(high, 256) * hist[high:]) / float(w2)

            # Calculate the between-class variance
            variance = w0 * (mean0 - mean) ** 2 + w1 * (mean1 - mean) ** 2 + w2 * (mean2 - mean) ** 2

            # Update the optimal threshold values if the variance is greater than the maximum variance
            if variance > max_variance:
                max_variance = variance
                optimal_high = high
                optimal_low = low

    # Apply thresholding to the input image using the optimal threshold values
    binary = np.zeros(gray_img.shape, dtype=np.uint8)
    binary[gray_img < optimal_low] = 0
    binary[(gray_img >= optimal_low) & (gray_img < optimal_high)] = 128
    binary[gray_img >= optimal_high] = 255

    return binary


def Thresholding(array, high, low, thresh=0):
    arrayCopy = array.copy()
    if(thresh == 0):
        thresh = np.mean(arrayCopy.ravel()) - 5
    arrayCopy[arrayCopy >= thresh] = high
    arrayCopy[arrayCopy < thresh] = low
    print("SH", array.shape)
    return arrayCopy


def localThresholding(image, size):
    result = np.zeros(image.shape)
    i = 0
    j = 0
    imgX = image.shape[1]
    imgY = image.shape[0]
    nX = size[0]
    nY = size[1]
    while(j < image.shape[1]):
        i = 0
        nX = size[0]
        while(i < image.shape[0]):
            result[i:nX, j:nY] = Thresholding(image[i:nX, j:nY], 255, 0,)
            i = nX
            nX += size[0]
        j = nY
        nY += size[1]
    return result

def spectral_thresholding_local(image, size):
    result = np.zeros(image.shape)
    i = 0
    j = 0
    imgX = image.shape[1]
    imgY = image.shape[0]
    nX = size[0]
    nY = size[1]
    while(j < image.shape[1]):
        i = 0
        nX = size[0]
        while(i < image.shape[0]):
            result[i:nX, j:nY] = spectral_thresholding(image[i:nX, j:nY])
            i = nX
            nX += size[0]
        j = nY
        nY += size[1]
    return result

def otsu_threshold_local(image, size):
    result = np.zeros(image.shape)
    i = 0
    j = 0
    imgX = image.shape[1]
    imgY = image.shape[0]
    nX = size[0]
    nY = size[1]
    while(j < image.shape[1]):
        i = 0
        nX = size[0]
        while(i < image.shape[0]):
            result[i:nX, j:nY] = otsu_threshold(image[i:nX, j:nY])[0]
            i = nX
            nX += size[0]
        j = nY
        nY += size[1]
    return result

def iterative_thresholding_local(image, size):
    result = np.zeros(image.shape)
    i = 0
    j = 0
    imgX = image.shape[1]
    imgY = image.shape[0]
    nX = size[0]
    nY = size[1]
    while(j < image.shape[1]):
        i = 0
        nX = size[0]
        while(i < image.shape[0]):
            result[i:nX, j:nY] = iterative_thresholding(image[i:nX, j:nY])[0]
            i = nX
            nX += size[0]
        j = nY
        nY += size[1]
    return result