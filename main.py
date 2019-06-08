from models.image import Image
from utils.file_utils import FileHelper

f = FileHelper('./files')
image_1 = f.load_key_points('test.png')
image_2 = f.load_key_points('test2.png')

x = image_1.nearest_keypoints_indexes(image_2)
print(x)