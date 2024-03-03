import cv2
import numpy as np


class ImageFractalDimension:
    def __init__(self, image):
        # image = Image.open(image_name)
        self.SIZE = image.shape[0]

        self.threshold = 140
        _, binary_image = cv2.threshold(image, self.threshold, 255, cv2.THRESH_BINARY)


        image_bw = np.asarray(binary_image)
        self.img_px_array = np.copy(image_bw)

        self.fractal_dim = self.calculate_fractal_dim()

    def calculate_fractal_dim(self):
        self.dimensions = []
        self.filled_boxes = []

        size = 1
        while size != self.SIZE:
            size *= 2
            filled_box = self.boxcount(size)
            self.filled_boxes.append(filled_box)
            self.dimensions.append(size / self.SIZE)

        return -np.polyfit(np.log(self.dimensions), np.log(self.filled_boxes), 1)[0]

    def blockshaped(self, square_size):
        h, w = self.img_px_array.shape
        assert h % square_size == 0, "Array is not evenly divisible".format(h, square_size)
        return (self.img_px_array.reshape(h // square_size, square_size, -1, square_size).swapaxes(1, 2).reshape(-1,
                                                                                                                 square_size,
                                                                                                                 square_size))
    def boxcount(self, size):
        blocked_arrays = self.blockshaped(size)
        counter = 0

        for i in range(len(blocked_arrays)):
            for j in range(len(blocked_arrays[i])):
                if (blocked_arrays[i][j].any()):
                    counter += 1
                    break
        return counter


