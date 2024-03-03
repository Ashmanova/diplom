import numpy as np
import cv2

import exponent_heldera
import fractal_dimension
import projective_transformations
import multifractal_spectr




if __name__ == '__main__':
    # считываем изображение
    image_path = "image/1.jpg"
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # сжимаем изображение
    small_image = cv2.resize(gray_image, (32, 32))

    # рассчитываем экспоненту Гельдера

    # matrix_heldera = exponent_heldera.exp_heldera(small_image)
    # np.savetxt('matrix_heldera.txt', matrix_heldera, fmt='%.18e')
    # matrix_heldera = np.loadtxt('matrix.txt')
    # print(matrix_heldera)

    # разделяем изображение
    # matrix_region = exponent_heldera.divide_into_regions(matrix_heldera)
    # print(matrix_region)
    # np.savetxt('matrix_region.txt', matrix_region, fmt='%d')

    # рассчитываем мультифрактальный спектор
    multifractal = multifractal_spectr.calculate_multifractal_spectr(small_image)
    print(multifractal)


    # frcatal_image = fractal_dimension.ImageFractalDimension(small_image)
    # fractal_dimension = frcatal_image.fractal_dim
    # print(fractal_dimension)
