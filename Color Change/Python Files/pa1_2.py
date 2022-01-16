import numpy as np
import cv2
import sys


def saveImage(image):
    cv2.imwrite("Pictures/" + "q= " + sys.argv[4] + " - " + sys.argv[3], image)


def find_quantized_value(old_pixel, qVal):
    return int(qVal * (old_pixel / 255)) * (255 / qVal)


def boundValue(pix_val, error, fractional):
    if pix_val + int(error * fractional) < 0:
        number = 0
    elif pix_val + int(error * fractional) > 255:
        number = 255
    else:
        number = pix_val + (error * fractional)

    return number


def FloydSteinberg(image, qVal):
    width, height = image.shape

    for y in range(height - 1):
        for x in range(width - 1):
            old_pixel = image[x][y]
            new_pixel = find_quantized_value(old_pixel, qVal)
            image[x][y] = new_pixel
            error = old_pixel - new_pixel
            image[x + 1][y] = (boundValue(image[x + 1][y], error, (7 / 16)))
            image[x - 1][y + 1] = (boundValue(image[x - 1][y + 1], error, (3 / 16)))
            image[x][y + 1] = (boundValue(image[x][y + 1], error, (5 / 16)))
            image[x + 1][y + 1] = (boundValue(image[x + 1][y + 1], error, (1 / 16)))
    saveImage(image)
    return image
