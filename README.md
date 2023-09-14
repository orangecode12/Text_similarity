Text Similarity Tool


This tool compares phrases from an input text with a list of standardized phrases to find similarities based on embeddings from the MiniLM model.

To install the necessary libraries, use: pip install -r requirements.txt   

To start the program type: python main.py "sample_text.txt"
Where "sample_text.txt" is text file you want to analyse.

The model used for embeddings is all-MiniLM-L6-v2 from the sentence_transformers library. The alternative model is bert-base-uncased. The usage of these models is similar,
but all-MiniLM-L6-v2 gives better results analyzing long ngrams. The best similarities found with both models are available accordingly in results-all-MiniLM-L6-v2.txt and results-bert-base-uncased.txt. 

This script generates 2 to 16-grams from the input text and computes their embeddings. It then calculates the cosine similarities between these embeddings and the embeddings of the standardized terms. Phrases with cosine similarities above the threshold (0.6 for MiniLM) are considered similar and are included in the output.

Further research. To improve the output it is possible to test other models. It must be useful to create more data for testing and validating scripts.