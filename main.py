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


def getHeuristic(heuristic_name, low_r, high_r, min_neighbours):
    if heuristic_name == 'random':
        return RandomHeuristic()
    elif heuristic_name == 'distance':
        return DistanceHeuristic(low_r, high_r)
    elif heuristic_name == 'probability':
        return ProbabilityHeuristic()
    elif heuristic_name == 'reduction':
        return ReductionHeuristic()
    elif heuristic_name == 'neighbours':
        return NeighbourHeuristic(min_neighbours)
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

    if not os.path.exists(dest):
        os.makedirs(dest)

    os.system(f'cp {filepath} {dest}/')
    os.system(f'mv {filepath}.* {dest}/')

def run(file, file2, dest=None, extract_points=True,
        neighbours_limit=50, threshold=0.5, max_error=15,
        iterations=5000, pairs=None, consistent_pairs=None,
        transformation='affine', heuristic_choice='random',
        ransac=True, low_r=4, high_r=400, min_neighbours=5, p=0.5):

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

    print(f'Pairs count: {len(pairs)}')
    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_pairs.png', lines=[pairs], orientation='horizontal')

    if consistent_pairs is None:
        consistent_pairs = pairs_processor.consistent_pairs(pairs, neighbours_limit, threshold)
        file_helper.save_as_json(f'{common_name}_consistent_pairs.json', consistent_pairs)
        con_pairs = file_helper.load_from_json(f'{common_name}_consistent_pairs.json')
    else:
        con_pairs = file_helper.load_from_json(consistent_pairs)

    print(f'Consistent pairs count: {len(con_pairs)}')
    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_consistent_pairs.png', lines=[con_pairs], orientation='horizontal')

    if ransac:
        it_est = len(con_pairs) / len(pairs)

        heuristic = getHeuristic(heuristic_choice, low_r, high_r, min_neighbours)
        transform = getTransformation(transformation)
        r = Ransac()

        result_pairs = r.start(con_pairs, max_error, iterations, transformation=transform, heuristic=heuristic, p=p, w=it_est)
        file_helper.save_as_json(f'{common_name}_ransac_pairs.json', result_pairs)

        ransac_pairs = file_helper.load_from_json(f'{common_name}_ransac_pairs.json')
        print(f'Ransac pairs count: {len(ransac_pairs)}')

        img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_ransac_pairs.png', colors=[(255, 255, 0, 255)], lines=[ransac_pairs], orientation='horizontal')
        img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_all_pairs.png', lines=[pairs, con_pairs, ransac_pairs], orientation='horizontal')


def _make_images(dest, file, file2, pairs=None, consistent_pairs=None, ransac_pairs=None):
    file_helper = FileHelper(dest)
    img_helper = ImageHelper(dest)

    pairs = file_helper.load_from_json(pairs)
    print(f'Pairs: {len(pairs)}')
    common_name = get_common_substring(file, file2)

    img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_clean_out.png', lines=[])

    if consistent_pairs is not None:
        consistent_pairs = file_helper.load_from_json(consistent_pairs)
        print(f'Consistent pairs: {len(consistent_pairs)}')
        img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_pairs_conspairs_out.png', lines=[pairs, consistent_pairs], colors=[(0,0,255,255), (255,255,0,255)])

        if ransac_pairs is not None:
            ransac_pairs = file_helper.load_from_json(ransac_pairs)
            print(f'Ransac pairs: {len(ransac_pairs)}')
            img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_ransac_pairs_out.png',
                                  colors=[(255, 255, 0, 255)], lines=[ransac_pairs], orientation='horizontal')
            img_helper.draw_lines([f'{file}.png', f'{file2}.png'], f'{common_name}_pairs_ranscons_out.png',
                                  lines=[consistent_pairs, ransac_pairs], colors=[(0, 0, 255, 255), (255, 255, 0, 255)])

if __name__ == '__main__':
    # run('files/book/book_background.png', 'files/book/book_left.png',
    #     extract_points=True, dest='book/book_final_test_good', transformation='perspective', heuristic_choice='distance',
    #     low_r=4, high_r=400, p=0.7, max_error=20)
    #
    # run('files/book/book_background.png', 'files/book/book_left.png',
    #     extract_points=True, dest='book/book_final_test_bad', transformation='perspective', heuristic_choice='random',
    #     low_r=4, high_r=400, p=0.7, max_error=40, threshold=0.7, neighbours_limit=10)

    name = 'book_'
    _make_images('./files/book/book_final_test_good', 'book_background', 'book_left', pairs=f'{name}_pairs.json', consistent_pairs=f'{name}_consistent_pairs.json', ransac_pairs=f'{name}_ransac_pairs.json')

'''
        
        
'''