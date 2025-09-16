"""
split the data into train, dev, and test sets in the ratio of 80:10:10
"""
import os
import random
from tqdm import tqdm
import shutil
import locale


locale.setlocale(locale.LC_COLLATE, 'ko_KR.UTF-8')

def split_data(input_file, output_dir, train_ratio=0.8, dev_ratio=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read the file 
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # shuffle the lines
    random.shuffle(lines)

    # calculate the number of lines for each split
    total_lines = len(lines)
    train_size = int(total_lines * train_ratio)
    dev_size = int(total_lines * dev_ratio)

    # split the data
    train_data = lines[:train_size]
    dev_data = lines[train_size:train_size + dev_size]
    test_data = lines[train_size + dev_size:]

    # write the data to files
    with open(os.path.join(output_dir, 'train.tsv'), 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_data)

    with open(os.path.join(output_dir, 'dev.tsv'), 'w', encoding='utf-8') as dev_file:
        dev_file.writelines(dev_data)

    with open(os.path.join(output_dir, 'test.tsv'), 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_data)

    print(f"Data split into {len(train_data)} train, {len(dev_data)} dev, and {len(test_data)} test samples.")
    
    return output_dir


if __name__ == "__main__":
    input_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/processed/jje-kor_tokenised_filtered.tsv"
    output_dir = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/input"
    
    split_data(input_file, output_dir)
    print("data has been split into train, dev, and test sets.")