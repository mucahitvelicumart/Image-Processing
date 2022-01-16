import numpy as np
from matplotlib import pyplot as plt
from PIL import Image, ImageFilter


def MeanFilter(image, windowSize):
    width, height = image.size
    kernelValue = int((windowSize - 1) / 2)
    width = int(width - kernelValue)
    height = int(height - kernelValue)
    for i in range(kernelValue, width):
        for j in range(kernelValue, height):
            new_R, new_G, new_B = 0, 0, 0
            for k in range(-kernelValue, kernelValue + 1):
                for l in range(-kernelValue, kernelValue + 1):
                    new_R = new_R + image.getpixel((i + k, j + l))[0]
                    new_G = new_G + image.getpixel((i + k, j + l))[1]
                    new_B = new_B + image.getpixel((i + k, j + l))[2]
            new_R, new_G, new_B = int(new_R / (windowSize ** 2)), int(new_G / (windowSize ** 2)), int(
                new_B / (windowSize ** 2))
            image.putpixel((i, j), (new_R, new_G, new_B))
    return image


def GaussianFilter(image, windowSize, sigma):
    rangeVal = int((windowSize - 1) / 2)

    gaussian_mask = np.zeros((windowSize, windowSize))
    for i in range(-rangeVal, rangeVal + 1):
        for j in range(-rangeVal, rangeVal + 1):
            var_x = sigma ** 2 * (2 * np.pi)
            var_y = np.exp(-(i ** 2 + j ** 2) / (2 * sigma ** 2))
            gaussian_mask[i + rangeVal, j + rangeVal] = (1 / var_x) * var_y

    width, height = image.size
    width = int(width - rangeVal)
    height = int(height - rangeVal)
    for i in range(rangeVal, width):
        for j in range(rangeVal, height):
            new_R, new_G, new_B = 0, 0, 0
            for k in range(-rangeVal, rangeVal + 1):
                for l in range(-rangeVal, rangeVal + 1):
                    new_R = (new_R + (image.getpixel((i + k, j + l))[0] * gaussian_mask[k + rangeVal][l + rangeVal]))
                    new_G = (new_G + (image.getpixel((i + k, j + l))[1] * gaussian_mask[k + rangeVal][l + rangeVal]))
                    new_B = (new_B + (image.getpixel((i + k, j + l))[2] * gaussian_mask[k + rangeVal][l + rangeVal]))
            image.putpixel((i, j), (int(new_R), int(new_G), int(new_B)))
    return image


def KuwaharaFilter(image_path, filter_dimension):
    RGB_image = np.array(Image.open(image_path), dtype=float)
    HSV_image = np.array(Image.open(image_path).convert("HSV"), dtype=float)
    image_width, image_height, image_channel = HSV_image.shape
    padding_value = filter_dimension // 2
    for row in range(padding_value, image_width - padding_value):
        for column in range(padding_value, image_height - padding_value):
            image_part = HSV_image[row - padding_value: row + padding_value + 1,
                         column - padding_value: column + padding_value + 1, 2]
            width, height = image_part.shape
            Q1 = image_part[0: height // 2 + 1, width // 2: width]
            Q2 = image_part[0: height // 2 + 1, 0: width // 2 + 1]
            Q3 = image_part[height // 2: height, 0: width // 2 + 1]
            Q4 = image_part[height // 2: height, width // 2: width]
            stds = np.array([np.std(Q1), np.std(Q2), np.std(Q3), np.std(Q4)])
            min_std = stds.argmin()

            if min_std == 0:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row - height // 2: row + 1, column: column + width // 2 + 1, 2])

            elif min_std == 1:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row - height // 2: row + 1, column - width // 2: column + 1, 2])

            elif min_std == 2:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column - width // 2: column + 1, 2])

            elif min_std == 3:
                RGB_image[row][column][0] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 0])
                RGB_image[row][column][1] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 1])
                RGB_image[row][column][2] = np.mean(
                    RGB_image[row: row + height // 2 + 1, column: column + height // 2 + 1, 2])

    return Image.fromarray(RGB_image.astype(np.uint8))


if __name__ == '__main__':
    for var in [3, 5, 7, 9]:
        # Example 1
        img = Image.open('Pictures/balloon.jpg')
        img = MeanFilter(img, var)
        img.save("./Output/balloon_Mean_" + str(var) + ".jpg")
        img = Image.open('Pictures/balloon.jpg')
        img = GaussianFilter(img, var, 1)
        img.save("./Output/balloon_Gaussian_" + str(var) + ".jpg")
        img = KuwaharaFilter("./Pictures/balloon.jpg", var)
        img.save("./Output/balloon_Kuwahara_" + str(var) + ".jpg")
        img = Image.open('Pictures/balloon.jpg')
        img = GaussianFilter(img, var, 2)
        img.save("./Output/balloon_Gaussian_" + str(var) + "_sigma2.jpg")

        # Example 2
        img = Image.open('Pictures/france.jpg')
        img = MeanFilter(img, var)
        img.save("./Output/france_Mean_" + str(var) + ".jpg")
        img = Image.open('Pictures/france.jpg')
        img = GaussianFilter(img, var, 1)
        img.save("./Output/france_Gaussian_" + str(var) + ".jpg")
        img = KuwaharaFilter("./Pictures/france.jpg", var)
        img.save("./Output/france_Kuwahara_" + str(var) + ".jpg")
        img = Image.open('Pictures/france.jpg')
        img = GaussianFilter(img, var, 2)
        img.save("./Output/france_Gaussian_" + str(var) + "_sigma2.jpg")

        # Example 3
        img = Image.open('Pictures/colorful.jpg')
        img = MeanFilter(img, var)
        img.save("./Output/colorful_Mean_" + str(var) + ".jpg")
        img = Image.open('Pictures/colorful.jpg')
        img = GaussianFilter(img, var, 1)
        img.save("./Output/colorful_Gaussian_" + str(var) + ".jpg")
        img = KuwaharaFilter("./Pictures/colorful.jpg", var)
        img.save("./Output/colorful_Kuwahara_" + str(var) + ".jpg")
        img = Image.open('Pictures/colorful.jpg')
        img = GaussianFilter(img, var, 2)
        img.save("./Output/colorful_Gaussian_" + str(var) + "_sigma2.jpg")

        # Example 4
        img = Image.open('Pictures/lesson.jpg')
        img = MeanFilter(img, var)
        img.save("./Output/lesson_Mean_" + str(var) + ".jpg")
        img = Image.open('Pictures/lesson.jpg')
        img = GaussianFilter(img, var, 1)
        img.save("./Output/lesson_Gaussian_" + str(var) + ".jpg")
        img = KuwaharaFilter("./Pictures/lesson.jpg", var)
        img.save("./Output/lesson_Kuwahara_" + str(var) + ".jpg")
        img = Image.open('Pictures/lesson.jpg')
        img = GaussianFilter(img, var, 2)
        img.save("./Output/lesson_Gaussian_" + str(var) + "_sigma2.jpg")

        # Example 5
        img = Image.open('Pictures/self.jpg')
        img = MeanFilter(img, var)
        img.save("./Output/self_Mean_" + str(var) + ".jpg")
        img = Image.open('Pictures/self.jpg')
        img = GaussianFilter(img, var, 1)
        img.save("./Output/self_Gaussian_" + str(var) + ".jpg")
        img = KuwaharaFilter("./Pictures/self.jpg", var)
        img.save("./Output/self_Kuwahara_" + str(var) + ".jpg")
        img = Image.open('Pictures/self.jpg')
        img = GaussianFilter(img, var, 2)
        img.save("./Output/self_Gaussian_" + str(var) + "_sigma2.jpg")
