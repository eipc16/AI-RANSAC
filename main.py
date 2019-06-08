from models.image import Image
from utils.file_utils import FileHelper
from utils.image_utils import ImageHelper
from logic.pairs_processing import PairProcessor

file = 'coffee'

f = FileHelper(f'./files/{file}')
i = ImageHelper(f'./files/{file}')
p = PairProcessor()
image_1 = f.load_key_points(f'{file}1.png')
image_2 = f.load_key_points(f'{file}2.png')

x = image_1.get_keypoint_pairs(image_2)
cx = p.consistent_pairs(x, 10, 0.6)

json_x = Image.get_pairs_dict(x)
json_cx = Image.get_pairs_dict(cx)

f.save_as_json(f'{file}_pairs.json', json_x)
f.save_as_json(f'{file}_consistent_pairs.json', json_cx)

x = f.load_from_json(f'{file}_pairs.json')
cx = f.load_from_json(f'{file}_consistent_pairs.json')

i.draw_lines([f'{file}1.png', f'{file}2.png'], f'{file}_out.png', keypoint_pairs=x, consistent_keypoint_pairs=cx, orientation='horizontal')