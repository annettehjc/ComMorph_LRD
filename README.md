# ComMorph_LRD
Paper: Morphological Analysis on Jeju and Korean Using Transformer [(pdf)](Morphological_analysis_on_jeju_and_korean_using_transoformer_models.pdf) 
Author: Hyunjoo Cho [(email)](hyunjoo.cho@student.uni-tuebingen.de)

## Info
This repository contains the code framework used in the study to measure similarity between Korean and Jeju using encoder-decoder transformer models.

1. [**extract_data_from_original_data**](extract_data_from_original_data) contains files to process different data format to extract needed transcription sentences of Jeju and Korean.

2. [**create_organise_data**](create_organise_data) contains files to process data file for cleaning, and splitting them into training, validation, and test sets. 

3. [**train_model_evaluate**](train_model_evaluate) contains files to create vocabulary files, to train models with arguments, and to evaluate on them to see the performance level. 

The experiment is based on data from [jeju-dialect-to-standard](https://huggingface.co/datasets/junyeong-nero/jeju-dialect-to-standard), [jeju_potato_dataset](https://huggingface.co/datasets/jeju-potato/jeju_potato_datasets), and [jejueo_interview_transcripts](https://huggingface.co/datasets/mickeyshoes/jejueo_interview_transcripts). 
