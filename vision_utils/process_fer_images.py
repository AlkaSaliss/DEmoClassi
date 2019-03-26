import argparse
import pandas as pd
from vision_utils.face_utils import align_and_crop, select_list_images
from vision_utils.custom_torch_utils import processing_time
import warnings
warnings.filterwarnings('ignore')


@processing_time
def load_process_save_images(path_to_csv, class_, flag_, root_path, path_detector=None):
    df = pd.read_csv(path_to_csv, sep=',')
    list_images = select_list_images(df, class_, flag_)
    res = align_and_crop(list_images, class_=class_, flag=flag_, root_path=root_path, path_detector=path_detector)
    print(res)


ROOT_PATH = "../../fer2013/processed_images"
PATH_TO_CSV = "../../fer2013/fer2013.csv"
CLASS_ = list(range(7))
FLAGS = ['PublicTest', 'PrivateTest', "Training"]  # Training, PublicTest, PrivateTest


def main(args=None):
    if args is None:
        parser = argparse.ArgumentParser("Processing fer2013 images")
        parser.add_argument('--path_to_detector', type=str, default=PATH_TO_CSV,
                            help='Path to the dlib pretrained detector')
        parser.add_argument('--root_path', type=str, default=ROOT_PATH, help='Path to the root directory where' +
                                                                             'to save the processed images')
        parser.add_argument('--classes', nargs='+', type=int, default=CLASS_,
                            help='list of Class labels for which to process all related images')
        parser.add_argument('--flags', nargs='+', type=str, default=FLAGS,
                            help='list of Dataset split on which to apply ' +
                                 'the pre-processing, Training, PublicTest, PrivateTest ')
        args = parser.parse_args()

        for flag in args.flags:
            for cl in args.classes:
                print("####******* Processing class {} from {} Dataset *******####".format(cl, flag))
                load_process_save_images(args.path_to_csv, cl, flag, args.root_path, args.path_to_detector)
                print('\n\n')
    else:
        for flag in args['flags']:
            for cl in args['classes']:
                print("####******* Processing class {} from {} Dataset *******####".format(cl, flag))
                load_process_save_images(args['path_to_csv'], cl, flag, args['root_path'], args['path_detector'])
                print('\n\n')


if __name__ == '__main__':
    main()
