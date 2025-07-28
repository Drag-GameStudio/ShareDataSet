from __future__ import annotations
import cv2
from utils import liner_opacity
import numpy as np
import math

class ImageObject:
    def __init__(self, image=None):
        self.image = image
    
    def load_img(self, file_path):
        self.image = cv2.imread(file_path)

    def resize(self, size):
        self.image = cv2.resize(self.image, size)

    def rotate(self, angle):
        (h, w) = self.image.shape[:2]
        center = (w // 2, h // 2)

        scale = 1.0

        M = cv2.getRotationMatrix2D(center, angle, scale)
        self.image = cv2.warpAffine(self.image, M, (w, h))

    def get_window(self, pos, size):
        window = []
        for y_index in range(pos[1], min(self.image.shape[0], size[1] + pos[1])):
            current = self.image[y_index][pos[0]:pos[0] + size[0]]
            if current.shape[0] < size[0]:
                for _ in range(size[0] - current.shape[0]):
                    current = np.append(current, current[-1])
            window.append(current)
        if len(window) < size[1]:
            for _ in range(size[1] - len(window)):
                window.append(window[-1])
        return np.array(window)

    def adap_image(self, back_image: ImageObject, pos, pixels_smooth, smooth_fun):
        bg_pixels = back_image.get_window(pos, [self.image.shape[1], self.image.shape[0]])
        image_center = (self.image.shape[1] // 2, self.image.shape[0] // 2)
        max_radius = ((image_center[0] + image_center[1]) / 2) * 1.5

        for y_index in range(self.image.shape[0] - 1):
            for x_index in range(self.image.shape[1]):
                    curr_radius = math.sqrt(math.pow(image_center[0] - x_index, 2) + math.pow(image_center[1] - y_index, 2))

                    if curr_radius > max_radius - pixels_smooth:
                        self.image[y_index][x_index] = smooth_fun(bg_pixels[y_index][x_index], self.image[y_index][x_index], (max_radius - curr_radius) / max_radius)
                

    def add_image(self, new_image: ImageObject, position: tuple[int, int], size: tuple[int, int], smooth_fun, pixels_smooth):
        new_image.resize(size)
        new_image.adap_image(back_image=self, pos=position, pixels_smooth=pixels_smooth, smooth_fun=smooth_fun)

        pixels = new_image.image
        for y_index in range(position[1], self.image.shape[0]):
            if y_index < size[1] + position[1]:
                for x_index in range(position[0], self.image.shape[1]):
                    if x_index < size[0] + position[0]:
                        self.image[y_index][x_index] = pixels[y_index - position[1]][x_index - position[0]]
            
if __name__ == "__main__":
    back = ImageObject()
    face = ImageObject()

    back.load_img(r"back.jpg")
    face.load_img(r"face.jpg")

    back.add_image(face, (0, 0), (100, 100), liner_opacity, 40)
    cv2.imshow("r", back.image)
    cv2.waitKey(2000)