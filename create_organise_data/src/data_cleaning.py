"""
using Kkma from konlpy to do morphological tokenisation of Jeju-Korean parallel corpus.
"""
from konlpy.tag import Kkma 
from tqdm import tqdm

kkma = Kkma()

def tokeniser(input_file, output_file):
    analyzers = kkma

    file = open(input_file, 'r', encoding='utf-8')
    lines = file.readlines()

    # if the line includes the markers, ignore them
    checked_lines = []
    for line in tqdm(lines, desc="Filtering lines"):
        if not any(keyword in line for keyword in ['name', 'business_name', 'place', 'address', 'bank', 'person', 'church', 'program_name', 'word', 'product_name']):
            checked_lines.append(line)

    tokenised_lines = []
    for line in tqdm(checked_lines, desc="Tokenising lines"):
        # separate the text by tab and get first as jeju and second as kor
        jeju, kor = line.split('\t')
        # if there was a problem, and only jeju has the ending punctuation, add it to kor
        if jeju.endswith('.') and not kor.endswith('.'):
            kor += '.'
        if jeju.endswith('!') and not kor.endswith('!'):
            kor += '!'
        if jeju.endswith('?') and not kor.endswith('?'):
            kor += '?'
        # if the puctuation is not added anywhere
        if not (jeju.endswith('.') or jeju.endswith('!') or jeju.endswith('?')):
            jeju += '.'
            kor += '.'

        # do morphological tokenisation 
        tkn_jeju = analyzers.morphs(jeju)
        tkn_kor = analyzers.morphs(kor)
        tokenised_jeju_kor = f"{tkn_jeju}\t{tkn_kor}"
        # remove unnecessary brackets, commas, and quotations
        tokenised_jeju_kor = tokenised_jeju_kor.replace("'", "").replace("[", "").replace("]", "").replace(",", "")

        tokenised_lines.append(tokenised_jeju_kor)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tokenised_lines:
            outfile.write(line + '\n')

    # print("Morphs")
    # morphs = analyzers.morphs(text)
    # print(morphs)

    # print("Nouns")
    # nouns = analyzers.nouns(text)
    # print(nouns)

    # print("Pos")
    # pos = analyzers.pos(text)
    # print(pos)

    # print("Sentences")
    # sent = analyzers.sentences(text)
    # print(sent)



if __name__ == "__main__":
    input_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/output/sorted_jeju-standard_parallel.tsv"
    # test_input_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/processed/jje-kor_small.tsv"
    output_file = "/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/output/preprocessed_jeju-standard_parallel.tsv"
    # output_file = "./test.tsv"
    tokeniser(input_file, output_file)
    print("file toknisation is done.")
