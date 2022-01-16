import sys
import cv2
import numpy as np
import pa1_2 as FS


def readGrayScaleImage():
    return cv2.imread("Pictures/" + sys.argv[1], 0)


def saveImage(image):
    cv2.imwrite("Pictures/" + "q= " + sys.argv[4] + " - " + (sys.argv[2]), image)


def quantizeImage(image, qVal):
    height, width = image.shape

    for i in range(height):
        for j in range(width):
            image[i][j] = int(qVal * (image[i][j] / 255)) * (255 / qVal)
    saveImage(image)
    return image


def showImage(image, text):
    cv2.imshow(text, image)


if __name__ == "__main__":
    q_value = int(sys.argv[4])
    img = readGrayScaleImage()
    img = cv2.resize(img, (320, 250))
    showImage(img, "Original Image")
    img = quantizeImage(img, q_value)
    showImage(img, "Quantized Image")

    img2 = readGrayScaleImage()
    img2 = FS.FloydSteinberg(img2, q_value)
    img2 = cv2.resize(img2, (320, 250))
    showImage(img2, "FloydSteinberg Image")
    cv2.waitKey()
    cv2.destroyAllWindows()
