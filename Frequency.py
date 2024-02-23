import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_filter(X, H):
    # make sure both X and H are 2-D
    assert(X.ndim == 2)
    assert(H.ndim == 2)

    # get the horizontal and vertical size of X and H
    imageColumns = X.shape[1]
    imageRows = X.shape[0]
    kernelColumns = H.shape[1]
    kernelRows = H.shape[0]

    # calculate the horizontal and vertical size of Y (assume "full" convolution)
    newRows = imageRows + kernelRows - 1
    newColumns = imageColumns + kernelColumns - 1

    # create an empty output array
    Y = np.zeros((newRows, newColumns))

    # go over output locations
    for m in range(newRows):
        for n in range(newColumns):
            # go over input locations
            for i in range(kernelRows):
                for j in range(kernelColumns):
                    if (m-i >= 0) and (m-i < imageRows) and (n-j >= 0) and (n-j < imageColumns):

                        Y[m, n] = Y[m, n] + H[i, j]*X[m-i, n-j]
        # make sure kernel is within bounds

    return Y

def high_pass(image):
    filter = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    filterd = apply_filter(image, filter)

    cv2.imwrite('debug/original_high.jpg', image)
    cv2.imwrite('debug/filterd_high.jpg', filterd)

    return filterd

def low_pass(image):
    filter = (1/16) * np.array([[1, 4, 1],
                                [4, 16, 4],
                                [1, 4, 1]])
    filterd = apply_filter(image, filter)

    cv2.imwrite('debug/original_low.jpg', image)
    cv2.imwrite('debug/filterd_low.jpg', filterd)

    return filterd

def hypird_image (image1,image2):

    image1=cv2.resize(image1,(400,300))
    image2=cv2.resize(image2,(400,300))


    low_freq=low_pass(image1)
    high_freq=high_pass(image2)
    result = low_freq+high_freq
    
    cv2.imwrite('debug/hypird.jpg', result)
   
    return result
    
def using_fourier(image1,image2):

    image1=cv2.resize(image1,(400,300))
    image2=cv2.resize(image2,(400,300))

    # apply fourier

    Fourier1 = np.fft.fftshift(image1)
    Fourier2 = np.fft.fftshift(image2)

    #  Getting  of certian radius
    M, N = Fourier1.shape
    circle = np.zeros((M, N), dtype=np.float32)
    radius = 50
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u-M/2)**2 + (v-N/2)**2)
            if D <= radius:
                circle[u, v] = 1
            else:
                circle[u, v] = 0

    # Filter: Low pass filter
    low_image = Fourier1 * circle

    # Filter: High pass filter
    circle = 1 - circle
    high_image = Fourier2 * circle

    result = low_image +high_image
    # Inverse Fourier Transform
    # result = np.fft.ifftshift(result)
    result = np.abs(np.fft.ifft2(result))
    cv2.imwrite('debug/original_high.jpg', image1)
    # cv2.imwrite('debug/filterd_high.jpg', filterd)
    cv2.imwrite('debug/original_low.jpg', image2)
    # cv2.imwrite('debug/filterd_low.jpg', filterd)
    cv2.imwrite('debug/hypird.jpg', result)

    return result


def create_gaussian_filter(h, w, sigma):
    center = int(h/2), int(w/2)
    Y = np.arange(h).reshape(-1, 1)
    X = np.arange(w).reshape(1, -1)
    # kernel = np.exp(- ((Y - center[0]) ** 2 + (X - center[1]) ** 2) / (2 * sigma**2)) / (2 * np.pi * (sigma ** 2))
    kernel = np.exp(- ((Y - center[0]) ** 2 + (X - center[1]) ** 2) / (2 * sigma**2))
    return kernel


def fft_low_pass(img,sigma):
    # img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    kernel = create_gaussian_filter(f.shape[0], f.shape[1], sigma)
    fft_image = np.fft.fftshift(f) * kernel 
    inversed = np.fft.ifft2(np.fft.ifftshift(fft_image))
    inversed = np.abs(inversed)
    # inversed = (inversed - inversed.min()) / inversed.max()
    cv2.imwrite('debug/low_gauss.jpg', inversed)
    print('done')
    return fft_image,inversed

def fft_high_pass(img,sigma):
    # img = cv2.imread(image_path, flags=cv2.IMREAD_GRAYSCALE)
    f = np.fft.fft2(img)
    kernel = 1-create_gaussian_filter(f.shape[0], f.shape[1], sigma)
    fft_image = np.fft.fftshift(f) * kernel 
    inversed = np.fft.ifft2(np.fft.ifftshift(fft_image))
    inversed = np.abs(inversed)
    # inversed = (inversed - inversed.min()) / inversed.max()
    cv2.imwrite('debug/high_gauss.jpg', inversed)
    print('done')
    return fft_image,inversed
    

# image1 = cv2.imread('Images\\star.png', 0)
# image2 = cv2.imread('Images/Everest_North_Face_toward_Base_Camp_Tibet_Luca_Galuzzi_2006.jpg', 0)

def fft_hyprid_image(img1,img2,sigma):


    image1=cv2.resize(img1,(400,300))
    image2=cv2.resize(img2,(400,300))

    cv2.imwrite('debug/original_low.jpg' , image1)
    cv2.imwrite('debug/original_high.jpg', image2)

    fimg1,img1=fft_low_pass(image1,sigma)
    fimg2,img2=fft_high_pass(image2,sigma)

    inversed = np.fft.ifft2(np.fft.ifftshift(fimg1+fimg2))
    inversed = np.abs(inversed)

    cv2.imwrite('debug/lgauss.jpg', img1)
    cv2.imwrite('debug/hgauss.jpg', img2)
    cv2.imwrite('debug/gauss.jpg', inversed)
    return img1, img2, img1+img2