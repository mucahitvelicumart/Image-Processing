import sys
import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import pa2_2

if __name__ == "__main__":
    sourceImage = skimage.io.imread("Pictures2/" + sys.argv[1])
    targetImage = skimage.io.imread("Pictures2/" + sys.argv[2])
    fig = plt.figure(figsize=(30, 10))
    fig.add_subplot(1, 3, 1)
    plt.imshow(sourceImage)
    plt.title('Source')
    plt.axis('off')

    fig.add_subplot(1, 3, 2)
    plt.imshow(targetImage)
    plt.title('Target')
    plt.axis('off')

    resultImage = pa2_2.colorTransfer(sourceImage, targetImage)
    fig.add_subplot(1, 3, 3)
    plt.imshow(resultImage)
    plt.title('Result')
    plt.axis('off')

    plt.show()
