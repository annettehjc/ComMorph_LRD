"""
to get the word from Jeju potato dataset
https://huggingface.co/datasets/jeju-potato/jeju_potato_datasets/viewer/default/train?row=3&views%5B%5D=train
in the local computer, it's path is: ./ComMorph_LRD/data/labeling_data
"""
import os
import json
from tqdm import tqdm

def get_sentences(dir, output_file):
    labeling_data_path = os.path.abspath(dir)

    # iterate through all JSON files in the folder, and make a list of them 
    json_files = [f for f in os.listdir(labeling_data_path) if f.endswith(".json")]
    # json_to_process = json_files[:10]  # limit to first 10 files for processing, to check the output format

    # create an output file to write the setences
    with open(output_file, "w", encoding="utf-8") as out_f:
        # for tqdm progress bar
        for filename in tqdm(json_files, desc="Processing files"):
            # open each file and read the content
            with open(os.path.join(labeling_data_path, filename), "r", encoding="utf-8") as in_f:
                data = json.load(in_f)
                # Extract the transcription field to get nested dialect and standard sentences 
                transcription = data.get("transcription", {})
                dialect = transcription.get("dialect")
                standard = transcription.get("standard")
                
                # if both dialect and standard sentences are present, write them to the output file
                if dialect and standard:
                    item = dialect + "\t" + standard + "\n"
                    out_f.write(item)

    return output_file

if __name__ == "__main__":
    # data path
    data_path = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/jeju_potato-labeling_data/split_07"
    output_file = "output/standard-jeju_parallel_7.tsv"

    result_file = get_sentences(data_path, output_file)
    print(f"Output saved to: {result_file}")