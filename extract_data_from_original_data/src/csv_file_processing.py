"""
to get the data from Jeju_Standard.csv file. 
Extract exact two columns: dialect_form and standard_form 
make a tsv file with these two columns in a format of dialect_form \t standard_form
"""
import pandas as pd

# load the CSV file
df = pd.read_csv("/Users/annette/Desktop/Germany/U_of_Tuebingen/4th-Semester(SS24)/Computational Morphology/project/ComMorph_LRD/data/Jeju_Standard.csv")

# keep only the two columns
df_selected = df[["dialect_form", "standard_form"]]

# save as tsv
df_selected.to_csv("jeju_standard.tsv", sep="\t", index=False, header=False)
