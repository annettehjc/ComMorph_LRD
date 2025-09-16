"""
to get the word from Jeju potato dataset
https://huggingface.co/datasets/jeju-potato/jeju_potato_datasets/viewer/default/train?row=3&views%5B%5D=train
in the local computer, it's path is: ./ComMorph_LRD/data/labeling_data
"""
import os
import json

def get_dialects(dir):
    # Path to the labeling_data folder
    labeling_data_path = os.path.abspath(dir)

    all_dialects = []

    # Iterate through all folders in labeling_data
    for folder in os.listdir(labeling_data_path):
        folder_path = os.path.join(labeling_data_path, folder)
        if os.path.isdir(folder_path):
            # Iterate through all JSON files in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".json"):
                    file_path = os.path.join(folder_path, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        segments = data.get("transcription", {}).get("segments", [])
                        for segment in segments:
                            dialect = segment.get("dialect")
                            standard = segment.get("standard")
                            if dialect:
                                item = f"{dialect} ({standard})"
                                all_dialects.append(item)
    return all_dialects

def frequent_dialects(dialects):
    # go through the lists of dialects and count the frequency of each dialect. make dictionry out of it.
    dialect_count = {}
    for dialect in dialects:
        if dialect in dialect_count:
            dialect_count[dialect] += 1
        else:
            dialect_count[dialect] = 1
    # sort the dictionary by frequency
    sorted_dialects = sorted(dialect_count.items(), key=lambda item: item[1], reverse=True)
    return sorted_dialects

def ignore_none_type(dialects):
    return [item for item in dialects if "(None)" not in item[0]]

def main():
    # Specify the path to the labeling_data folder
    labeling_data_path = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/labeling_data"

    # Get all dialects
    dialects = get_dialects(labeling_data_path)

    # Print the list of dialects
    sorted_dialects = frequent_dialects(dialects)

    cleaned_dialects = ignore_none_type(sorted_dialects)
    
    return cleaned_dialects

if __name__ == "__main__":
    cleaned_dialects = main()

    # save output of main into a file 
    output_file = "frequent_dialects.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        for dialect, count in cleaned_dialects:
            f.write(f"{dialect}: {count}\n")
