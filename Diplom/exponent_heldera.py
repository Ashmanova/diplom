import math
from sklearn.linear_model import LinearRegression
import numpy as np



# с отражением пикселей
def get_color(image, x, y):
    nR, nC = image.shape
    x = abs(x) if x < 0 else (nR * 2 - x - 2) if x >= nR else x
    y = abs(y) if y < 0 else (nC * 2 - y - 2) if y >= nC else y
    return image[x, y]


def get_gradient(image, x, y):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])   # Оператор Собеля для производной по x
    kernel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])   # Оператор Собеля для производной по y
    # Вычисляем значения градиентов
    gx , gy = 0 , 0
    for i in range(3):
        for j in range(3):
            gx += kernel_x[i, j] * get_color(image, x - 1 + i, y - 1 + j)
            gy += kernel_y[i, j] * get_color(image, x - 1 + i, y - 1 + j)
    gx /= 255.0
    gy /= 255.0
    return (gx**2 + gy**2) / 255.0


def get_energy(image, x, y):
    return get_color(image,x,y)*get_gradient(image,x,y)


def exp_heldera(image, rmax=20):
    nR, nC = image.shape #Количесвто строк и столбцов
    res = np.zeros((nR, nC), dtype=float) # Сюда записываются результаты
    regX = np.array([t * 2 + 1 for t in range(rmax)])
    regY = np.zeros(rmax)
    for i in range(nR):
        for j in range(nC):
            # print(i,j)
            regY[0] = get_energy(image, i, j)
            for r in range(1,rmax):
                regY[r] = regY[r-1]
                for t in range(-r, r+1):
                    regY[r] += get_energy(image, i - r, j + t)
                    regY[r] += get_energy(image, i + r, j + t)
                    if t != -r and t != r:
                        regY[r] += get_energy(image, i + t, j - r)
                        regY[r] += get_energy(image, i + t, j + r)
            for r in range(rmax):
                regY[r] = math.log(1+regY[r])
                # Линейная регрессия с использованием scikit-learn
                X = regX.reshape(-1, 1)  # Преобразование вектора regX в двумерный массив
                y = regY.reshape(-1, 1)  # Преобразование вектора regY в двумерный массив
                model = LinearRegression().fit(X, y)
                res[i, j] = model.coef_[0, 0]  # Наклон прямой (slope) сохраняется в res[i, j]
    return res


def divide_into_regions(matrix):
    # Проверка, что матрица не пуста
    if not matrix.any():
        raise ValueError("Пустая матрица")

    thresholds = np.linspace(matrix.min(), matrix.max(), num=11)
    regions = np.zeros_like(matrix, dtype=np.uint8)

    for i in range(1, len(thresholds)):
        lower_threshold = thresholds[i - 1]
        upper_threshold = thresholds[i]
        regions[(matrix >= lower_threshold) & (matrix < upper_threshold)] = i

    return regions







