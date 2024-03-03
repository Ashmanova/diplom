import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
img = cv2.imread('C:\Users\22354\PycharmProjects\Diplom\image\1.jpg')

# Определение исходных искажающих точек
src_points = np.float32([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]])

# Определение конечных точек (пример)
dst_points = np.float32([[100, 50], [img.shape[1] - 100, 50], [50, img.shape[0] - 50], [img.shape[1] - 50, img.shape[0] - 50]])

# Вычисление матрицы преобразования
matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Применение преобразования
result = cv2.warpPerspective(img, matrix, (img.shape[1], img.shape[0]))

# Визуализация исходного и искаженного изображений
plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Transformed Image')
plt.show()
