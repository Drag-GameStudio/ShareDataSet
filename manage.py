from core import ImageObject
import random
from utils import liner_opacity
import cv2
import os
from py_progress.progress import ProgressBar

class Generator:
    def __init__(self, front_folder_images, back_folder_images):
        self.front_folder_images = front_folder_images
        self.back_folder_images = back_folder_images

    def init_folders(self):
        self.front_images_path = []
        for dirpath, dirnames, filenames in os.walk(self.front_folder_images):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                self.front_images_path.append(full_path)
                
        self.back_images_path = []
        for dirpath, dirnames, filenames in os.walk(self.back_folder_images):
            for filename in filenames:
                full_path = os.path.join(dirpath, filename)
                self.back_images_path.append(full_path)
                

    def generate_one_sample(self, front_image, back_image, with_rotation: bool = False):
        front = ImageObject(front_image)
        back = ImageObject(back_image)

        if with_rotation:
            front.rotate(random.randint(0, 15))

        x_size = int(max(back.image.shape[0] * 0.8, random.random() *  back.image.shape[0]))
        size = (x_size, int((x_size * front.image.shape[0]) / front.image.shape[1]))
        pos = (int(random.random() * (back.image.shape[1] - size[0] - 1)), int(random.random() * (back.image.shape[0] - size[1] - 1)))

        pixels_smooth = round((max(size[0], size[1]) / 2) * 0.9)
        back.add_image(front, position=pos, size=size, smooth_fun=liner_opacity, pixels_smooth=pixels_smooth)

        return back.image
    
    def generate_samples(self, count_per_one, exit_folder=None, with_rotation: bool = False):
        pb = ProgressBar(len(self.front_images_path))
        for front_image_path in self.front_images_path:
            front_image = cv2.imread(front_image_path)
            for i in range(count_per_one):
                back_image = cv2.imread(self.back_images_path[round(random.random() * (len(self.back_images_path) - 1))])
                mixed_image = self.generate_one_sample(front_image, back_image, with_rotation=with_rotation)

                if exit_folder is not None:
                    cv2.imwrite(f"{exit_folder}\\{front_image_path.split("\\")[-1].split(".")[0]}_{i}.png", mixed_image)
            pb.progress_with_time("")

if __name__ == "__main__":
    
    gen = Generator(r"C:\Users\huina\Python Projects\FaceDetector-master\default\humans", r"C:\Users\huina\Python Projects\Impotant projects\ShareDataSet\src\back")
    gen.init_folders()
    gen.generate_samples(2, r"C:\Users\huina\Python Projects\FaceDetector-master\alg_without_rot\humans", with_rotation=True)