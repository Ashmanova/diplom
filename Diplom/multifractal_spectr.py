import cv2
import numpy as np

import exponent_heldera
import fractal_dimension




def calculate_multifractal_spectr(image):
    # рассчитываем экспоненту Гельдера
    matrix_heldera = np.loadtxt('matrix.txt')

    # разделяем изображение
    matrix_region = exponent_heldera.divide_into_regions(matrix_heldera)

    # рассчитываем мультифрактальный спектор
    unique_labels = np.unique(matrix_region)
    fractal_dimensions = np.zeros(len(unique_labels))

    # Рассчет фрактальной размерности для каждой области
    for i, label in enumerate(unique_labels):
        if label!=0:
            print(label)
            mask = (matrix_region == label)
            mask_image = image.copy()
            mask_image[~mask] = 0
            cv2.imshow("", mask_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            frcatal_image = fractal_dimension.ImageFractalDimension(mask_image)
            fractal_dimensions[i] = frcatal_image.fractal_dim


    return (fractal_dimensions)