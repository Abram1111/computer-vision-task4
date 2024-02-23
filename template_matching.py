import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def Normalised_Cross_Correlation(roi, target):
    # Normalised Cross Correlation Equation
    cor =np.sum(roi*target)
    nor = np.sqrt((np.sum(roi**2)))*np.sqrt(np.sum(target**2))
    return cor / nor

def template_matching(img, target):
    # initial parameter
    height,width        =img.shape
    tar_height,tar_width=target.shape
    (max_Y,max_X)=(0, 0)
    MaxValue = 0

    # Set image, target and result value matrix
    img=np.array(img, dtype="int")
    target=np.array(target, dtype="int")
    NccValue=np.zeros((height-tar_height,width-tar_width))

    # calculate value using filter-kind operation from top-left to bottom-right
    for y in range(0,height-tar_height):
        for x in range(0,width-tar_width):
            # image roi
            roi =img[y:y+tar_height,x:x+tar_width]
            # calculate ncc value
            NccValue[y,x] = Normalised_Cross_Correlation(roi,target)
            
            # find the most match area
            if NccValue[y,x]>MaxValue:
                MaxValue=NccValue[y,x]
                (max_Y,max_X) = (y,x)

    return (max_X,max_Y)


def call_matching (img ,template):
    t0 = time.time()
    # Call the template matching function
    top_left_cord=template_matching(img, template)
    #Calculating remaining coordinates
    top_right_cord=(top_left_cord[0]+template.shape[1]-1 , top_left_cord[1])
    bottom_left   =(top_left_cord[0] , top_left_cord[1]+template.shape[0]-1)
    bottom_right  =(bottom_left[0] +template.shape[1]-1 , bottom_left[1])

    plt.figure(figsize=(15,15))

    plt.plot([top_left_cord[0],top_right_cord[0]], [top_left_cord[1],top_right_cord[1]],color="black", linewidth=3)
    plt.plot([bottom_left[0],bottom_right[0]], [bottom_left[1],bottom_right[1]],color="black", linewidth=3)
    plt.plot([top_left_cord[0],bottom_left[0]],[top_left_cord[1],bottom_left[1]],color='black', linewidth=3)
    plt.plot([top_right_cord[0],bottom_right[0]],[top_right_cord[1],bottom_right[1]],color='black', linewidth=3)
    plt.imshow(img ,cmap='gray')

    plt.savefig('template/result.png')
    t1 = time.time()
    return t1-t0



def SSD(img, template):
    t0 = time.time()
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    img_height, img_width = gray_img.shape
    temp_height, temp_width = template.shape
   
    
    score = np.empty((img_height-temp_height, img_width-temp_width))
  
    #h[m,n]=sum ((g[k,l]- f [m+k,n+l])**2)
    for dy in range(0, img_height - temp_height):
        for dx in range(0, img_width - temp_width):
            diff = (gray_img[dy:dy + temp_height, dx:dx + temp_width] - template)**2
            score[dy, dx] = diff.sum()
    min = np.unravel_index(score.argmin(), score.shape)
    point=(min[1], min[0])
    cv2.rectangle(img, (point[0], point[1] ), (point[0] + temp_width, point[1] + temp_height), (0,0,200), 3)
    t1 = time.time()
    cv2.imwrite("template/ssd.png", img)
    return t1-t0