from utils.file_utils import FileHelper
from utils.image_utils import ImageHelper
from logic.pairs_processing import PairProcessor
from heuristics.heuristics import *
from transformations.transform import AffineTransformation, PerspectiveTransformation
from logic.ransac import Ransac
import os

def get_common_substring(left_string, right_string):
    if not (isinstance(left_string, str) and isinstance(right_string, str)):
        return None

    output = ''

    for i, j in zip(left_string, right_string):
        if i == j:
            output += i
        else:
            break

    return output


def getHeuristic(heuristic_name, low_r=4, high_r=300):
    if heuristic_name == 'random':
        return RandomHeuristic()
    elif heuristic_name == 'distance':
        return DistanceHeuristic(low_r, high_r)
    elif heuristic_name == 'probability':
        return ProbabilityHeuristic()
    elif heuristic_name == 'reduction':
        return ReductionHeuristic()
    else:
        return None

def getTransformation(transformation_name):
    if transformation_name == 'affine':
        return AffineTransformation()
    elif transformation_name == 'perspective':
        return PerspectiveTransformation()
    else:
        return None


def extract_filename_from_path(path):
    file = path.split("/")[-1]
    filename = file.split(".")[0]
    return filename

def run_extractor(filepath, dest):
    os.system(f'./extractor/extract_features_64bit.ln -haraff -sift -i {filepath} -DE')
    filename = extract_filename_from_path(filepath)
    os.system(f'cp {filepath} {dest}/')
    os.system(f'mv {filepath}.* {dest}/')

def run(file, file2, dest=None, extract_points=True,
        neighbours_limit=5, threshold=0.4, max_error=20,
        iterations=1000, pairs=None, consistent_pairs=None,
        transformation='affine', heuristic_choice='random'):

    filepath, filepath2 = file, file2
    file, file2 = extract_filename_from_path(file), extract_filename_from_path(file2)
    common_name = get_common_substring(file, file2)
    if dest is None:
        dest = f'./files/{common_name}'
    else:
        dest = f'./files/{dest}'

    if not os.path.exists(dest):
        os.makedirs(dest)

    if extract_points:
        run_extractor(filepath, dest)
        run_extractor(filepath2, dest)

    file_helper, img_helper, pairs_processor = FileHelper(dest), ImageHelper(dest), PairProcessor()

    if pairs is None:
        image_1, image_2 = file_helper.load_key_points(f'{file}.png'), file_helper.load_key_points(f'{file2}.png')
        point_pairs = image_1.get_keypoint_pairs(image_2)
        file_helper.save_as_json(f'{common_name}_pairs.json', point_pairs)
        pairs = file_helper.load_from_json(f'{common_name}_pairs.json')
    else:
        pairs = file_helper.load_from_json(pairs)

    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_pairs.png', lines=[pairs], orientation='horizontal')

    if consistent_pairs is None:
        consistent_pairs = pairs_processor.consistent_pairs(pairs, neighbours_limit, threshold)
        file_helper.save_as_json(f'{common_name}_consistent_pairs.json', consistent_pairs)
        con_pairs = file_helper.load_from_json(f'{common_name}_consistent_pairs.json')
    else:
        con_pairs = file_helper.load_from_json(consistent_pairs)

    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_consistent_pairs.png', lines=[con_pairs], orientation='horizontal')

    it_est = 1 - (len(pairs) / len(con_pairs))

    heuristic = getHeuristic(heuristic_choice)
    transform = getTransformation(transformation)
    r = Ransac()

    result_pairs = r.start(con_pairs, max_error, iterations, transformation=transform, heuristic=heuristic)
    file_helper.save_as_json(f'{common_name}_ransac_pairs.json', result_pairs)

    ransac_pairs = file_helper.load_from_json(f'{common_name}_ransac_pairs.json')

    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_ransac_pairs.png', lines=[ransac_pairs], orientation='horizontal')
    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_all_pairs.png', lines=[pairs, con_pairs, ransac_pairs], orientation='horizontal')


if __name__ == '__main__':
    # run('files/mug/mug1.png', 'files/mug/mug2.png', extract_points=True, dest='mug_persp_dist', transformation='perspective', heuristic_choice='distance')
    run('files/mug/mug1.png', 'files/mug/mug2.png', extract_points=True, dest='mug_persp_prob', transformation='perspective', heuristic_choice='reduction')
    # run('files/glasses/glasses2.png', 'files/glasses/glasses1.png', extract_points=True, dest='glasses_out', transformation='perspective', heuristic_choice='probability')