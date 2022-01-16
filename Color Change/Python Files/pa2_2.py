import sys
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from math import *


def replaceZeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data


def image_stats(image):
    mean = np.mean(image, axis=1)
    std = np.std(image, axis=1)
    return mean, std


def multiplyMatrices(a, b):
    result_of_multiply = np.dot(a, b)
    return result_of_multiply


def colorTransfer(source, target):
    ro = source.shape[0]
    co = source.shape[1]

    sourceImage = source / 255
    targetImage = target / 255

    # RGB Channels
    source_R, source_G, source_B = np.rollaxis(sourceImage, -1)
    target_R, target_G, target_B = np.rollaxis(targetImage, -1)

    r1 = source_R.reshape((source_R.shape[0] * source_R.shape[1], 1))
    g1 = source_G.reshape((source_G.shape[0] * source_G.shape[1], 1))
    b1 = source_B.reshape((source_B.shape[0] * source_B.shape[1], 1))

    r2 = target_R.reshape((target_R.shape[0] * target_R.shape[1], 1))
    g2 = target_G.reshape((target_G.shape[0] * target_G.shape[1], 1))
    b2 = target_B.reshape((target_B.shape[0] * target_B.shape[1], 1))

    sourceImage = np.hstack((r1, g1, b1))
    targetImage = np.hstack((r2, g2, b2))

    # Numpy array for Matrices
    a = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    b = np.array([[1 / np.sqrt(3), 0, 0], [0, 1 / np.sqrt(6), 0], [0, 0, 1 / np.sqrt(2)]])
    c = np.array([[1, 1, 1], [1, 1, -2], [1, -1, 0]])
    b2 = np.array([[np.sqrt(3) / 3, 0, 0], [0, np.sqrt(6) / 6, 0], [0, 0, np.sqrt(2) / 2]])
    c2 = np.array([[1, 1, 1], [1, 1, -1], [1, -2, 0]])

    # Step1
    source_LMS = multiplyMatrices(a, sourceImage.T)
    target_LMS = multiplyMatrices(a, targetImage.T)

    # Step2
    source_LMS = replaceZeroes(source_LMS)
    source_LOG = np.where(source_LMS > 0.0000000001, np.log10(source_LMS), -10)

    target_LMS = replaceZeroes(target_LMS)
    target_LOG = np.where(target_LMS > 0.0000000001, np.log10(target_LMS), -10)

    # Step3
    p1 = multiplyMatrices(b, c)
    source_LAB = multiplyMatrices(p1, source_LOG)
    target_LAB = multiplyMatrices(p1, target_LOG)

    # Step 4
    s_mean, s_std = image_stats(source_LAB)
    t_mean, t_std = image_stats(target_LAB)

    sf = t_std / s_std

    result_LAB = np.zeros(source_LAB.shape)

    # Step 5-6-7
    for ch in range(0, 3):
        result_LAB[ch, :] = (source_LAB[ch, :] - s_mean[ch]) * sf[ch] + t_mean[ch]

    # Step 8
    result_LMS = multiplyMatrices(multiplyMatrices(c2, b2), result_LAB)

    # Step 9
    for ch in range(0, 3):
        result_LMS[ch, :] = np.power(10, result_LMS[ch, :])

    # Step 10
    d = np.array([[4.4679, -3.5873, 0.1193], [-1.2186, 2.3809, -0.1624], [0.0497, -0.2439, 1.2045]])
    est_im = multiplyMatrices(d, result_LMS).T
    result = est_im.reshape((ro, co, 3));

    return result
