import argparse
import re
from os import listdir, makedirs
from os.path import isfile, join, exists
from shutil import copyfile

# File name
file_name_re = r'.*\/(.*)'
dir_name_re = r'(.*\/).*'
remove_softmax_re = r'(.*)_SM.*\.(res|spt)'
ending_re = r'.*\.(res|spt)'

def _merge_files(destination, source_1, source_2):
    if exists(destination):
        raise ValueError("{} already exists".format(destination))
    else:
        dir = re.sub(dir_name_re, r'\1', destination)
        if not exists(dir):
            makedirs(dir)
        with open(source_1, 'r') as f1:
            first_text = f1.read()
        with open(source_2, 'r') as f2:
            if source_2.endswith('.res'):
                f2.readline()
            second_text = f2.read()

        first_text += second_text

        with open(destination, 'w') as d:
            d.write(first_text)

        print("Merged {} and {} into {}".format(source_1, source_2, destination))


def _copy_file(source, destination):
    if exists(destination):
        raise ValueError("{} already exists in {}".format(source, destination))
    else:
        dir = re.sub(dir_name_re, 'r\1', destination)
        if not exists(dir):
            makedirs(dir)
        copyfile(source, destination)


def _merge_results(sources, destination, **kwargs):
    if len(sources) != 2:
        raise ValueError("You must specify exactly two source directories")
    if destination in sources:
        raise ValueError("You cannot use source directory as destination")

    source_1_files = [join(sources[0], f) for f in listdir(sources[0]) if f[-4:] in ['.res', '.spt']]
    source_2_files = [join(sources[1], f) for f in listdir(sources[1]) if f[-4:] in ['.res', '.spt']]

    to_match_1 = [(re.sub(remove_softmax_re, r'\1.\2', re.sub(file_name_re, r'\1', f)), f) for f in source_1_files]
    to_match_2 = [(re.sub(remove_softmax_re, r'\1.\2', re.sub(file_name_re, r'\1', f)), f) for f in source_2_files]

    handled_1 = [False] * len(to_match_1)
    handled_2 = [False] * len(to_match_2)
    to_merge = []

    # Merging lists
    for f1_idx, f1 in enumerate(to_match_1):
        for f2_idx, f2 in enumerate(to_match_2):
            if f1[0] == f2[0]:
                to_merge.append((f1, f2))
                handled_1[f1_idx] = True
                handled_2[f2_idx] = True
                break

    print(handled_1)
    print(handled_2)

    # Collecting unmerged
    to_copy = []
    for i, val in enumerate(handled_1):
        if not val:
            to_copy.append(to_match_1[i])
    for i, val in enumerate(handled_2):
        if not val:
            to_copy.append(to_match_2[i])

    print("Merge size: ", len(to_merge))
    print("Copy size: ", len(to_copy))

    for m in to_merge:
        print(m[0][0], m[1][0])

    print("########")
    to_copy.sort()
    for m in to_copy:
        print(m[0])

    print("########")

    # Performing merge
    for t in to_merge:
        dest = join(destination, t[0][0])
        print(dest)
        _merge_files(dest, t[0][1], t[1][1])

    # Performing copy
    for t in to_copy:
        dest = join(destination, t[0])
        _copy_file(t[1], dest)

    assert 2*len(to_merge) + len(to_copy) == len(source_1_files) + len(source_2_files)


def _main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Merge scores from different directories into new directory by appending')

    parser.add_argument('-s', '--sources', type=str, nargs=2,
                        help='Sources to merge')
    parser.add_argument('-d', '--destination', type=str,
                        help='The destination directory to add the files to')
    args = parser.parse_args()

    # Call config selection with gathered arguments
    _merge_results(**vars(args))


if __name__ == '__main__':
    _main()
