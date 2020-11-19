from argparse import ArgumentParser
from shutil import copy2
import os
import glob

"""
This script copies the .json + .png files from a dataset directory to a single directory removing redundant file names.
"""


def copy_dataset(source, destination_folder):
    count = 1
    for src_file_png in glob.iglob(source + '/**/*.png', recursive=True):
        file_name_png = src_file_png.split('/')[-1]
        folder_path = src_file_png[0:-len(file_name_png)]

        file_name_json = file_name_png[0:-4] + '.json'

        file_json = os.path.join(folder_path, file_name_json)
        file_png = os.path.join(folder_path, file_name_png)

        if os.path.exists(file_json):
            copy2(file_png, os.path.join(destination_folder, str(count) + '.png'))
            copy2(file_json, os.path.join(destination_folder, str(count) + '.json'))
            count += 1


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--source", "-s", type=str, required=True)
    arg_parser.add_argument("--destination", "-d", type=str, required=True)
    args = arg_parser.parse_args()
    copy_dataset(args.source, args.destination)
