import numpy as np
import cv2
import matplotlib.pyplot as plt
import function as functions


def add_sp_noise(image, ratio):
    x, y = image.shape
    g = np.zeros((x, y), dtype=np.float32)
    pepper = ratio
    salt = 1-pepper
    for i in range(x):
        for j in range(y):
            rand = np.random.random()
            if rand < pepper:
                g[i][j] = 0
            elif rand > salt:
                g[i][j] = 1
            else:
                g[i][j] = image[i][j]
    return g


def add_gaussian_noise(image, sigma=.3):
    x, y = image.shape
    mean = 0
    n = np.random.normal(loc=mean, scale=sigma, size=(x, y))
    g = image+n
    return g


def add_uniform_noise(image, ratio=.3):
    x, y = image.shape
    n = np.random.uniform(low=0, high=ratio, size=(x, y))
    g = image+n
    return g


def avrage_filter(img, dimention):
    x, y = img.shape
    kernal = np.ones((dimention, dimention), np.float32)/(dimention*dimention)
    result = np.zeros((x, y), dtype=np.float32)
    result = cv2.filter2D(img, -1, kernal)
    return result


def avrage_filter2(img: np.array, dimention=3):
    kernel = np.ones((dimention, dimention), np.float32)/(dimention*dimention)
    return functions.convolve(img, kernel)


def gaussian_filter(img: np.array, dimention=3, sigma=.1):
    kernel = functions.generate_gaussian_kernal(dimention, sigma)
    return functions.convolve(img, kernel)


def median_filter(img: np.array, dimention=3):
    x, y = img.shape
    convolved_img = np.zeros(shape=(x-2*dimention, y-2*dimention))
    for i in range(x-2*dimention):
        for j in range(y-2*dimention):
            mat = img[i:i+dimention, j:j+dimention]
            convolved_img[i, j] = np.median(mat)

    return convolved_img


def laplacian_edge(img: np.array):
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])
    return abs(functions.convolve(img, kernel))


def Prewitt_edge(img: np.array):
    kernel_x = np.array([[-1, 0, 1],
                         [-1, 0, 1],
                         [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1],
                         [0, 0, 0],
                         [1, 1, 1]])
    img_x = functions.convolve(img, kernel_x)
    img_y = functions.convolve(img, kernel_y)
    final_img = np.sqrt(img_x**2+img_y**2)
    return final_img


def Sobel_edge(img: np.array, x_dir, y_dir):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1],
                         [0, 0, 0],
                         [-1, -2, -1]])
    img_x = functions.convolve(img, kernel_x)
    img_y = functions.convolve(img, kernel_y)
    if x_dir and y_dir:
        final_img = np.sqrt(img_x*img_x+img_y*img_y)
        return abs(final_img)
    elif x_dir:
        return abs(img_x)
    elif y_dir:
        return abs(img_y)
    else:
        return img


def roberts_edge(img: np.array):
    kernel_x = np.array([[1, 0],
                         [0, -1],
                         ])
    kernel_y = np.array([[0, 1],
                         [-1, 0],
                         ])
    img_x = functions.convolve(img, kernel_x)
    img_y = functions.convolve(img, kernel_y)
    final_img = np.sqrt(img_x*img_x+img_y*img_y)
    return abs(final_img)


def canny_edge(img: np.array):
    gaussian_kernal = functions.generate_gaussian_kernal(10, 1)
    gaussian_img = functions.convolve(img, gaussian_kernal)
    img_x = Sobel_edge(gaussian_img, 1, 0)
    img_y = Sobel_edge(gaussian_img, 0, 1)
    mag = np.hypot(img_x, img_y)
    mag = mag/mag.max()*255
    theta = np.arctan2(img_y, img_x)
    supp = functions.non_max_suppression(mag, theta)
    res, week, str = functions.canny_threshold(supp)
    g = functions.hysteresis(res, week, str)
    return g
