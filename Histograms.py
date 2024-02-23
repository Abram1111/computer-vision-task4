import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly_express as px


def histEqualization(array, max_val):
    arrayFlat = array.ravel()
    histogram = get_histogram_array(arrayFlat, max_val+1)
    # print('histogram',histogram[:50])
    cdf = cumulative(histogram, array.shape)
    # print("cdf", cdf)
    norm = cdf * max_val
    # print('norm', norm)
    normalized = np.rint(norm).astype('int')
    # print('normalized', normalized)
    # print(normalized.shape)
    # print(array.shape)
    # result = arrayFlat[normalized]
    result = normalized[arrayFlat]
    return result


def get_histogram_array(array, max_val):
    hist = np.zeros(max_val)
    for pixel in array:
        hist[pixel] += 1
    return hist


def cumulative(array, shape):
    if(len(shape) > 1):
        noPixels = shape[0] * shape[1]
        pdf = (array)/(noPixels)
    else:
        pdf = (array)/sum(array)
    # print('pdf', pdf)
    cdf = np.zeros(len(pdf))
    for i in range(1, len(pdf)):
        cdf[i] = pdf[i]+cdf[i-1]
    # cdf = np.cumsum(pdf)
    return cdf


def drawCumulative(data, title=''):
    count, bins_count = np.histogram(data, bins=256)
    print("Counts ", count.shape)
    data1 = cumulative(count, count.shape)
    # unique, counts = np.unique(data1, return_counts=True)
    # fig = plt.figure()
    # plt.plot(unique, data1[:len(unique)])
    fig = plt.figure()
    if(title == 'green'):
        plt.plot(bins_count[1:], data1, color='g')
        plt.legend("Cumulative Curve for Green")
    elif(title == 'red'):
        plt.plot(bins_count[1:], data1, color='r')
        plt.legend("Cumulative Curve for Red")
    else:
        plt.plot(bins_count[1:], data1, color='b')
        plt.legend("Cumulative Curve for Blue")
    # plt.hist(data1, 256, [0, 256])
    return fig


def drawCumulative1(data):
    bins = []
    for i in range(3):
        count, bins_count = np.histogram(data[..., i], bins=256)
        print("Counts ", count.shape)
        data1 = cumulative(count, count.shape)
        bins.append([bins_count, data1])
    fig = plt.figure()
    fig = px.line(
        x=bins[0][0][1:],
        y=bins[0][1],
        color_discrete_sequence=['red'], labels={"x": "Gray Level", "y": "CDF"}
    )
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name'] = 'Cumulative curve of Red'
    fig.add_scatter(name="Cumulative curve of Green", x=bins[1][0][1:],
                    y=bins[1][1], line_color="green")
    fig.add_scatter(name="Cumulative curve of Blue", x=bins[2][0][1:],
                    y=bins[2][1], line_color="blue")
    return fig


def drawCumulativeEq(data, data2):
    count, bins_count = np.histogram(data, bins=256)
    count2, bins_count2 = np.histogram(data2, bins=256)
    data1 = cumulative(count, count.shape)
    data2 = cumulative(count2, count.shape)
    fig = px.line(x=bins_count[1:],
                  y=data1,
                  color_discrete_sequence=['red'], labels={"x": "Gray Level", "y": "CDF"})
    fig['data'][0]['showlegend'] = True
    fig['data'][0]['name'] = 'Equalized '
    fig.add_scatter(name="Original", x=bins_count2[1:],
                    y=data2, line_color="green")
    return fig


def Normalize(image):
    maxIntensity = max(image.ravel())
    minIntensity = min(image.ravel())
    copy = image.copy()
    copy = (copy-minIntensity)/(maxIntensity-minIntensity)
    copy *= 255
    return copy


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


def ToGrey(image):
    return np.dot(image[..., :3], [0.299, 0.587, 0.114])
