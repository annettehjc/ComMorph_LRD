"""
sort given file alphabetically to delete duplicates.
"""
from tqdm import tqdm
import locale

locale.setlocale(locale.LC_COLLATE, 'ko_KR.UTF-8')

def alpha_sort(input_file, output_file):
    # read the file 
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
            lines = text.splitlines()
            sorted_lines = sorted(lines, key=locale.strxfrm)
            for line in tqdm(sorted_lines, desc="Sorting lines"):
                outfile.write(line+'\n')
    return output_file
            



if __name__ == "__main__":
    input_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/output/final_jeju-standard_parallel.tsv"
    # test_input_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/output/test.tsv"
    output_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/output/sorted_jeju-standard_parallel.tsv"
    alpha_sort(input_file, output_file)
    # alpha_sort(test_input_file, output_file)
    print("data has been sorted alaphabetically.")
