from cv2 import pencilSketch, sqrt
import numpy as np 
import cv2
import matplotlib.pyplot as plt
from scipy import signal
from Sift import*
import time

############
#  SSD     #
############
def calculateSSD(desc_image1, desc_image2):
    ssd = 0.0
    if len(desc_image1) != len(desc_image2):
        # Return -1 to indicate error (features have different sizes)
        return -1
    for i in range(len(desc_image1)):
        ssd += (desc_image1[i] - desc_image2[i])**2
    final_ssd = np.sqrt(ssd)
    return final_ssd

def SSD(descriptor1, descriptor2, threshold):
    KeyPoints1 = descriptor1.shape[0]
    KeyPoints2 = descriptor2.shape[0]
    matches = []
    for kp1 in range(KeyPoints1):
        best_ssd = float('inf')
        best_index = -1
        for Kp2 in range(KeyPoints2):
            ssd = calculateSSD(descriptor1[kp1], descriptor2[Kp2])
            if ssd < best_ssd:
                best_ssd = ssd
                best_index = Kp2
        if best_ssd <= threshold:
            feature = cv2.DMatch()
            # The index of the feature in the first image
            feature.queryIdx = kp1
            # The index of the feature in the second image
            feature.trainIdx = best_index
            # The distance between the two features
            feature.distance = best_ssd
            matches.append(feature)
    return matches

#################################
# normalized_cross_corelation   #
#################################    
def calculate_NCC(desc_image1, desc_image2):
    difference1=(desc_image1 - np.mean(desc_image1))
    difference2=(desc_image2 - np.mean(desc_image2))
    correlation_vector = np.multiply(difference1, difference2)
 
    normlized_output = correlation_vector / np.sqrt(difference1**2,difference2**2)
    NCC = float(np.mean(normlized_output))


    return NCC

def normalized_cross_corelation(key_pts_1, key_pts_2, desc1, desc2, threshold):
    
    matches = []

    for index1 in range(len(desc1)):
        for index2 in range(len(desc2)):
            out1_norm = (desc1[index1] - np.mean(desc1[index1])) / (np.std(desc1[index1]))
            out2_norm = (desc2[index2] - np.mean(desc2[index2])) / (np.std(desc2[index2]))
            corr_vector = np.multiply(out1_norm, out2_norm)
            corr = float(np.mean(corr_vector))
            # only taking above threshold
            if corr > threshold:
                matches.append([index1, index2, corr])

    output = []
    for index in range(len(matches)):
        dis = np.linalg.norm(np.array(key_pts_1[matches[index][0]].pt) - np.array(key_pts_2[matches[index][1]].pt))
        output.append(cv2.DMatch(matches[index][0], matches[index][1], dis))
    return output


#########################
# Deriving function     #
#########################

def call_matching (original_img ,original_template,method,threshold,resize,Sift_builttIn):
    t0 = time.time()
    if resize:
            original_img = cv2.resize(original_img,(150,150))
            original_template = cv2.resize(original_template,(150,150))
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    template = cv2.cvtColor(original_template, cv2.COLOR_BGR2GRAY)

    if Sift_builttIn:
        sift = cv2.SIFT_create() 

        key_pts1, descriptor1 = sift.detectAndCompute(img, None)
        key_pts2, descriptor2 = sift.detectAndCompute(template, None)
    else:

        key_pts1, descriptor1 = computeKeypointsAndDescriptors(img)
        key_pts2, descriptor2 = computeKeypointsAndDescriptors(template)

    if method=="Normalized cross corelation":
        matches = normalized_cross_corelation(key_pts1,key_pts2,descriptor1, descriptor2, threshold/100)
    elif method=="SSD":
        matches = SSD(descriptor1, descriptor2, threshold)
    print(matches)
    img_matches = cv2.drawMatches(original_img, key_pts1, original_template, key_pts2, matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    img_matches = cv2.resize(img_matches,(500,300))

    cv2.imwrite('template/result.png', img_matches)

    t1 = time.time()
    return t1-t0






