"""
get tsv files, read them, and combine them into one tsv file.
"""
import os

def combine_tsv_files(input_dir, output_file):
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # iterate through all files in the input directory
        for filename in os.listdir(input_dir):
            if filename.endswith('.tsv'):
                file_path = os.path.join(input_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as data:
                    # read the content of the TSV file and write it to the output file
                    for line in data:
                        outfile.write(line)
    return output_file


if __name__ == "__main__":
    input_directory = "output"  # directory containing the TSV files
    output_file_path = "output/final_standard-jeju_parallel.tsv"  # output file path

    combined_file = combine_tsv_files(input_directory, output_file_path)
    print(f"combined tsv file has been created.")