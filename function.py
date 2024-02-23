import numpy as np
import cv2
import matplotlib.pyplot as plt


def read_img(path):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img_gray


def mean(image):
    img_mean = cv2.mean(image)
    return img_mean[0]


def segment(image, threshold):
    binary_image = np.copy(np.array(image))
    for i in range(len(binary_image)):
        for j in range(len(binary_image[0])):
            if binary_image[i][j] > threshold:
                binary_image[i][j] = 255
            else:
                binary_image[i][j] = 0
    return


def convolve(img: np.array, kernel: np.array) -> np.array:
    x, y = img.shape
    k = kernel.shape[0]
    convolved_img = np.zeros(shape=(x-2*k, y-2*k))
    for i in range(x-2*k):
        for j in range(y-2*k):
            mat = img[i:i+k, j:j+k]
            convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

    return convolved_img


def generate_gaussian_kernal(dimention, sigma):
    kernal = np.zeros(shape=(dimention, dimention))
    x = dimention//2
    if(dimention % 2 == 0):
        y = x
    else:
        y = x+1
    for i in range(-x, y):
        for j in range(-x, y):
            part1 = 1/(2*(np.pi)*(sigma**2))
            part2 = (i**2)+(j**2)/(2*(sigma**2))
            part3 = np.exp(-part2)
            kernal[i+x][j+x] = part1*part3
    return kernal


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):

            # angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            # angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            # angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            # angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i, j] >= q) and (img[i, j] >= r):
                Z[i, j] = img[i, j]
            else:
                Z[i, j] = 0

    return Z


def canny_threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(20)
    strong = np.int32(255)
    for i in range(M):
        for j in range(N):
            if img[i, j] >= highThreshold:
                res[i, j] = strong
            elif (img[i, j] <= highThreshold) and (img[i, j] >= lowThreshold):
                res[i, j] = weak

    return (res, weak, strong)


def hysteresis(img, weak, strong=255):
    M, N = img.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i, j] == weak):

                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                    or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0

    return img
