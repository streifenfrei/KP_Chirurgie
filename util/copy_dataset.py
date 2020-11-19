import os
from argparse import ArgumentParser
from shutil import copy2

"""
This script copies the .json files from a dataset directory to a single directory removing redundant file names.
"""


def copy_dataset(source, destination):
    file_counter = {}
    for root, _, files in os.walk(source):
        for file in files:
            file_name, file_extension = os.path.splitext(file)
            if file_extension == '.json':
                dest_file = file
                if file not in file_counter.keys():
                    file_counter[file] = 0
                else:
                    file_name += '_{0}'.format(str(file_counter[file]))
                    file_counter[file] += 1
                    dest_file = file_name + file_extension
                copy2(os.path.join(root, file), os.path.join(destination, dest_file))


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument("--source", "-s", type=str, required=True)
    arg_parser.add_argument("--destination", "-d", type=str, required=True)
    args = arg_parser.parse_args()
    copy_dataset(args.source, args.destination)
