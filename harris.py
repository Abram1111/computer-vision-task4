from libs import *


def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.2):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the derivative of the image in x and y direction
    dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)

    # Compute the elements of the Harris matrix
    Ixx = cv2.pow(dx, 2)
    Ixy = cv2.multiply(dx, dy)
    Iyy = cv2.pow(dy, 2)

    # Compute the sum of the elements in the local neighborhood of each pixel
    Sxx = cv2.boxFilter(Ixx, -1, (block_size, block_size))
    Sxy = cv2.boxFilter(Ixy, -1, (block_size, block_size))
    Syy = cv2.boxFilter(Iyy, -1, (block_size, block_size))

    # Compute the determinant and trace of the Harris matrix
    det = cv2.subtract(cv2.multiply(Sxx, Syy), cv2.pow(Sxy, 2))
    trace = cv2.add(Sxx, Syy)

    # Compute the Harris response for each pixel
    harris = cv2.divide(det, trace + k)

    # Threshold the Harris response
    harris = cv2.threshold(harris, threshold *
                           harris.max(), 255, cv2.THRESH_BINARY)[1]

    # Convert the Harris response to a uint8 image
    harris = np.uint8(harris)

    # Find the coordinates of the corners
    coords = np.column_stack(np.where(harris > 0))

    return coords
