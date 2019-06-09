from models.image import Image
from utils.file_utils import FileHelper
from utils.image_utils import ImageHelper
from utils.time_utils import get_execution_time
from logic.pairs_processing import PairProcessor
from heuristics.heuristics import RandomHeuristic
from transformations.transform import AffineTransformation
from logic.ransac import Ransac


def get_common_substring(left_string, right_string):
    if not (isinstance(left_string, str) and isinstance(right_string, str)):
        return None

    char_set_1 = set(left_string)
    char_set_2 = set(right_string)

    return "".join(char_set_1 & char_set_2)

def get_extension(filename):
    filename_parts = filename.split(".")

    if len(filename_parts) < 1:
        return None

    return filename_parts[-1]

def run(file, file2, dest=None, pairs=None, consistent_pairs=None):
    if not dest:
        dest = f'./files/{get_common_substring(file, file2)}'

    file_helper, img_helper, pairs_processor = FileHelper(dest), ImageHelper(dest), PairProcessor()
    file_ext, file2_ext = get_extension(file), get_extension(file2)

    if file_ext != file2_ext:
        throw


file = 'mug'

f = FileHelper(f'./files/{file}')
i = ImageHelper(f'./files/{file}')
p = PairProcessor()

image_1 = f.load_key_points(f'{file}1.png')
image_2 = f.load_key_points(f'{file}2.png')

# x = image_1.get_keypoint_pairs(image_2)
# f.save_as_json(f'{file}_pairs.json', x)

x = f.load_from_json(f'{file}_pairs.json')

cx = p.consistent_pairs(x, 5, 0.6)
f.save_as_json(f'{file}_consistent_pairs.json', cx)

cx = f.load_from_json(f'{file}_consistent_pairs.json')

i.draw_lines([f'{file}1.png', f'{file}2.png'], f'{file}_out.png', keypoint_pairs=cx, consistent_keypoint_pairs=[], orientation='horizontal')

max_error = 20
it_count = 1000

h = RandomHeuristic()
t = AffineTransformation(h)
r = Ransac()

res = r.start(cx, max_error, it_count, t)

f.save_as_json(f'{file}_ransac_pairs.json', res)

rc = f.load_from_json(f'{file}_ransac_pairs.json')

i.draw_lines([f'{file}1.png', f'{file}2.png'], f'{file}_ransac_out.png', keypoint_pairs=[], consistent_keypoint_pairs=rc, orientation='horizontal')

